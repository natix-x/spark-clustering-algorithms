package clustering.algorithms.kmeans

import clustering.core.{Clusterer, Columns, EuclideanGeometry, Geometry}
import clustering.distance.DistanceMetric
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.storage.StorageLevel
import org.log4s.getLogger

import scala.util.Random

/** Breathing k-means (Fritzke, 2020–2023) — the `refine: breathing` knob on slot 1, NOT a
 *  separate algorithm: it is an outer add/remove cycle around the same [[LloydKMeans]] iteration
 *  that plain [[KMeans]] runs, so the two are a controlled comparison.
 *
 *  Escapes Lloyd's local minima by temporarily growing to k+m centroids (inserting near the
 *  highest-error ones), re-converging, then shrinking back to k (dropping the lowest-utility
 *  ones, with a freeze rule to avoid removing a close pair together). Keeps the result if it
 *  improved, otherwise shrinks m and retries from the best solution so far; stops at m = 0.
 *  Full write-up in the thesis.
 *
 *  @param m0       initial number of centroids added/removed per cycle (paper default 5)
 *  @param maxIter  maximum Lloyd iterations per phase
 *  @param maxCycles safety bound on breathing cycles; the paper's own bound is m reaching 0
 */
class BreathingKMeans(
  val k: Int,
  val m0: Int = 5,
  val maxIter: Int = 100,
  val eps: Double = 1e-4,
  val seed: Long = 42L,
  val geometry:Geometry = EuclideanGeometry,
  val maxCycles: Int = 100
) extends Clusterer {

  require(k >= 1, s"k must be >= 1, got $k")
  require(m0 >= 1, s"m0 must be >= 1, got $m0")

  /** Insertion offset scale, ε in the paper. */
  private val epsilon = 0.01

  private val logger = getLogger

  override def fit(data: DataFrame): KMeansModel = {
    val setup = LloydKMeans.initialize(data, geometry, k, seed, StorageLevel.MEMORY_AND_DISK)

    var bestCentroids = LloydKMeans.run(setup.preparedPoints, setup.initialCentroids, setup.fitDistance, geometry, maxIter, eps)
    var bestSSE = computeTotalError(setup.preparedPoints, bestCentroids, setup.fitDistance)
    logger.info(f"breathing: k=$k m0=$m0 initial SSE=$bestSSE%.4f")

    val random = new Random(seed)
    var currentM = m0
    var cycles = 0

    while (currentM > 0 && cycles < maxCycles) {
      // ── breathe in ────────────────────────────────────────────────────────────────
      val centroidStatistics = calculateCentroidStats(setup.preparedPoints, bestCentroids, setup.fitDistance)
      val totalMass = centroidStatistics.map(_.mass).sum
      val rmse = if (totalMass > 0.0) math.sqrt(centroidStatistics.map(_.error).sum / totalMass) else 0.0
      // At most one new centroid per existing centroid, so the breath is capped at |C| — with
      // k < m0 (e.g. k = 1) fewer than m centroids actually get inserted.
      val highestErrorCentroidsIndices = centroidStatistics.zipWithIndex.sortBy { case (s, i) => (-s.error, i) }.take(currentM).map(_._2)
      val newCentroids = highestErrorCentroidsIndices.map(i => generateOffset(bestCentroids(i), rmse, random))

      val expandedCentroids = LloydKMeans.run(setup.preparedPoints, bestCentroids ++ newCentroids, setup.fitDistance, geometry, maxIter, eps)

      // ── breathe out ───────────────────────────────────────────────────────────────
      // Remove exactly as many as were newly inserted: the invariant is "shrink back to k", not
      // "remove m". Removing m when fewer were inserted would empty the centroid set.
      val removalCount = expandedCentroids.length - k
      val reducedCentroids =
        if (removalCount <= 0) expandedCentroids
        else {
          val expandedStatistics = calculateCentroidStats(setup.preparedPoints, expandedCentroids, setup.fitDistance)
          val keptIndices = expandedCentroids.indices.toSet --
            selectCentroidsForRemoval(expandedCentroids, expandedStatistics, removalCount, setup.fitDistance)
          LloydKMeans.run(setup.preparedPoints, keptIndices.toArray.sorted.map(expandedCentroids), setup.fitDistance, geometry, maxIter, eps)
        }

      val candidateError = computeTotalError(setup.preparedPoints, reducedCentroids, setup.fitDistance)
      if (candidateError < bestSSE) {
        bestCentroids = reducedCentroids
        bestSSE = candidateError
      } else {
        // A failed cycle costs one unit of breath; the search restarts from the best solution so far.
        currentM -= 1
      }
      cycles += 1
      logger.info(f"breathing: cycle=$cycles m=$currentM SSE=$candidateError%.4f best=$bestSSE%.4f")
    }

    if (cycles >= maxCycles)
      logger.warn(s"breathing: stopped at the maxCycles=$maxCycles bound with m=$currentM")

    setup.release()
    new KMeansModel(bestCentroids, geometry.modelDistance)
  }

  /** `c + ε · RMSE · u`, u uniform in the unit hypercube centred at the origin. The result is
   *  projected onto the geometry, so on the unit sphere the inserted centroid stays a unit vector. */
  private def generateOffset(centroid: Vector, rmse: Double, random: Random): Vector = {
    val coordinates = centroid.toArray
    val perturbedCoordinates = new Array[Double](coordinates.length)
    var i = 0
    while (i < coordinates.length) {
      perturbedCoordinates(i) = coordinates(i) + epsilon * rmse * (random.nextDouble() - 0.5)
      i += 1
    }
    geometry.project(Vectors.dense(perturbedCoordinates))
  }

  /** Indices of the `count` centroids to delete: lowest utility first, freezing the nearest
   *  neighbour of every centroid picked, as long as `|frozen| + count < |C|`. */
  private def selectCentroidsForRemoval(
    candidateCentroids: Array[Vector],
    centroidStatistics: Array[BreathingKMeans.Stats],
    removalCount: Int,
    metric: DistanceMetric
  ): Set[Int] = {
    val utilityRanking = centroidStatistics.zipWithIndex.sortBy { case (s, i) => (s.utility, i) }.map(_._2)
    val removedIndices = scala.collection.mutable.Set.empty[Int]
    val protectedIndices = scala.collection.mutable.Set.empty[Int]

    utilityRanking.foreach { idx =>
      if (removedIndices.size < removalCount && !protectedIndices.contains(idx)) {
        removedIndices += idx
        if (protectedIndices.size + removalCount < candidateCentroids.length)
          findNearestNeighborIndex(candidateCentroids, idx, metric).foreach(protectedIndices += _)
      }
    }
    // A pathological freeze pattern could leave fewer than `removalCount` removals; fall back to
    // the plain utility ranking for the remainder so the set always shrinks back to k.
    if (removedIndices.size < removalCount)
      utilityRanking.foreach(i => if (removedIndices.size < removalCount) removedIndices += i)
    removedIndices.toSet
  }

  private def findNearestNeighborIndex(centroids: Array[Vector], idx: Int, metric: DistanceMetric): Option[Int] = {
    var nearestIndex = -1
    var nearestDistance = Double.MaxValue
    var candidateIndex = 0
    while (candidateIndex < centroids.length) {
      if (candidateIndex != idx) {
        val d = metric.compute(centroids(idx), centroids(candidateIndex))
        if (d < nearestDistance) { nearestDistance = d; nearestIndex = candidateIndex }
      }
      candidateIndex += 1
    }
    if (nearestIndex >= 0) Some(nearestIndex) else None
  }

  /** φ(C, X) = Σ_x w · d(x, nearest centroid)² — the objective breathing minimises. */
  private def computeTotalError(points: DataFrame, centroids: Array[Vector], metric: DistanceMetric): Double =
    calculateCentroidStats(points, centroids, metric).map(_.error).sum

  /** Per-centroid statistics needed by breathing k-means, all three in ONE Spark job.
   *
   *  For every point: `d1` is the distance to its nearest centroid, `d2` to its second nearest.
   *  Then, per centroid i with Voronoi set C_i (Fritzke's notation, weighted here):
   *    - `mass`  = Σ_{x∈C_i} w                    — how much data the centroid holds
   *    - `error` = Σ_{x∈C_i} w · d1²              — φ(c_i), the breathe-in criterion
   *    - `utility` = Σ_{x∈C_i} w · (d2² − d1²)    — U(c_i) = φ(C∖{c_i}) − φ(C), the breathe-out
   *      criterion: exactly the error increase caused by deleting c_i, since its points would
   *      fall back on their second-nearest centroid.
   *
   *  Breathing-only, hence private here rather than in [[LloydKMeans]] — no other centroid-based
   *  algorithm in this package needs it.
   */
  private def calculateCentroidStats(
    points: DataFrame,
    centroids: Array[Vector],
    distance: DistanceMetric
  ): Array[BreathingKMeans.Stats] = {
    val sc = points.sparkSession.sparkContext
    val broadcastCentroids = sc.broadcast(centroids)
    val dist = distance

    // (nearest index, d1², d2²) per point; d2 = +inf when there is a single centroid.
    val calculatePointDistancesUDF = udf { features: Vector =>
      val currentCentroids = broadcastCentroids.value
      var nearestIndex = 0
      var nearestDistance = Double.MaxValue
      var secondNearestDistance = Double.MaxValue
      var i = 0
      while (i < currentCentroids.length) {
        val d = dist.compute(features, currentCentroids(i))
        if (d < nearestDistance) {
          secondNearestDistance = nearestDistance;
          nearestDistance = d;
          nearestIndex = i
        }
        else if (d < secondNearestDistance) {
          secondNearestDistance = d
        }
        i += 1
      }
      val secondNearestSq =
        if (secondNearestDistance == Double.MaxValue) nearestDistance * nearestDistance
        else secondNearestDistance * secondNearestDistance
      (nearestIndex, nearestDistance * nearestDistance, secondNearestSq)
    }

    val aggregatedData = points
      .select(calculatePointDistancesUDF(col(Columns.Features)).as("s"), col(Columns.Weight))
      .select(
        col("s._1").as("centroid"),
        (col(Columns.Weight) * col("s._2")).as("error"),
        (col(Columns.Weight) * (col("s._3") - col("s._2"))).as("utility"),
        col(Columns.Weight).as("mass"))
      .groupBy("centroid")
      .agg(sum("mass").as("mass"), sum("error").as("error"), sum("utility").as("utility"))
      .collect()

    broadcastCentroids.unpersist(blocking = false)

    val metricsArray = Array.fill(centroids.length)(BreathingKMeans.Stats(0.0, 0.0, 0.0))
    aggregatedData.foreach { r =>
      metricsArray(r.getInt(0)) = BreathingKMeans.Stats(mass = r.getDouble(1), error = r.getDouble(2), utility = r.getDouble(3))
    }
    metricsArray
  }
}

private object BreathingKMeans {

  /** Per-centroid aggregates: `mass` = Σw, `error` = φ(c) = Σ w·d1², `utility` = U(c) = Σ w·(d2²−d1²). */
  private final case class Stats(mass: Double, error: Double, utility: Double)
}

