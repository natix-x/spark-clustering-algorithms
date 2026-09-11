package clustering.algorithms.kmeans

import clustering.core.{Clusterer, EuclideanGeometry, Geometry, Weights}
import clustering.utils.PartitionAggregator
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
    // Stats for `bestCentroids` are recomputed only when `bestCentroids` itself changes (a
    // successful cycle, below) — a FAILED cycle leaves `bestCentroids` untouched, so without this
    // cache the next "breathe in" step would re-run this full-data job on byte-identical input.
    // Cycles are expected to fail more often as `currentM` shrinks toward 0, so this removes a
    // real, growing fraction of the run's jobs, not just a one-off.
    var bestStats = calculateCentroidStats(setup.preparedPoints, bestCentroids, setup.fitDistance)
    var bestSSE = bestStats.map(_.error).sum
    logger.info(f"breathing: k=$k m0=$m0 initial SSE=$bestSSE%.4f")

    val random = new Random(seed)
    var currentM = m0
    var cycles = 0

    while (currentM > 0 && cycles < maxCycles) {
      // ── breathe in ────────────────────────────────────────────────────────────────
      val centroidStatistics = bestStats
      val totalMass = centroidStatistics.map(_.mass).sum
      // Scale of the insertion offset: a typical DISTANCE, so the mean cost has to be mapped
      // back through the geometry (euclidean = RMSE; spherical = mean 1 − cos, already a
      // distance). Taking sqrt unconditionally would offset by the square root of a distance.
      val offsetScale =
        if (totalMass > 0.0) geometry.costToDistance(centroidStatistics.map(_.error).sum / totalMass) else 0.0
      // At most one new centroid per existing centroid, so the breath is capped at |C| — with
      // k < m0 (e.g. k = 1) fewer than m centroids actually get inserted.
      val highestErrorCentroidsIndices = centroidStatistics.zipWithIndex.sortBy { case (s, i) => (-s.error, i) }.take(currentM).map(_._2)
      val newCentroids = highestErrorCentroidsIndices.map(i => generateOffset(bestCentroids(i), offsetScale, random))

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

      val reducedStats = calculateCentroidStats(setup.preparedPoints, reducedCentroids, setup.fitDistance)
      val candidateError = reducedStats.map(_.error).sum
      if (candidateError < bestSSE) {
        bestCentroids = reducedCentroids
        bestStats = reducedStats
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

  /** `c + ε · scale · u`, u uniform in the unit hypercube centred at the origin, `scale` the
   *  geometry's typical distance (Fritzke's RMSE under euclidean). The result is
   *  projected onto the geometry, so on the unit sphere the inserted centroid stays a unit vector. */
  private def generateOffset(centroid: Vector, scale: Double, random: Random): Vector = {
    val coordinates = centroid.toArray
    val perturbedCoordinates = new Array[Double](coordinates.length)
    var i = 0
    while (i < coordinates.length) {
      perturbedCoordinates(i) = coordinates(i) + epsilon * scale * (random.nextDouble() - 0.5)
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

  /** Plain nearest-neighbour scan (never top-2, so unlike the fold below the shrinking-bound
   *  early exit is safe here — see `calculateCentroidStats`'s comment for why top-2 is different).
   *  Raw-array + `distanceUpToOrdinal`: sqrt-free for Euclidean, and the bound genuinely shrinks
   *  as better candidates are found, same pattern as `NearestPrototypeModel.nearestRaw`. */
  private def findNearestNeighborIndex(centroids: Array[Vector], idx: Int, metric: DistanceMetric): Option[Int] = {
    val target = centroids(idx).toArray
    var nearestIndex = -1
    var nearestDistance = Double.MaxValue
    var candidateIndex = 0
    while (candidateIndex < centroids.length) {
      if (candidateIndex != idx) {
        val d = metric.distanceUpToOrdinal(target, centroids(candidateIndex).toArray, nearestDistance)
        if (d < nearestDistance) { nearestDistance = d; nearestIndex = candidateIndex }
      }
      candidateIndex += 1
    }
    if (nearestIndex >= 0) Some(nearestIndex) else None
  }

  /** Per-centroid statistics needed by breathing k-means, all three in ONE Spark job.
   *
   *  For every point: `d1` is the distance to its nearest centroid, `d2` to its second nearest.
   *  Then, per centroid i with Voronoi set C_i (Fritzke's notation, weighted here):
   *    - `mass`  = Σ_{x∈C_i} w                    — how much data the centroid holds
   *    - `error` = Σ_{x∈C_i} w · φ(d1)            — φ(c_i), the breathe-in criterion
   *    - `utility` = Σ_{x∈C_i} w · (φ(d2) − φ(d1)) — U(c_i) = φ(C∖{c_i}) − φ(C), the breathe-out
   *      criterion: exactly the error increase caused by deleting c_i, since its points would
   *      fall back on their second-nearest centroid. Note this stays exact under EITHER geometry
   *      precisely because both terms use the same φ — it is a difference of the objective with
   *      and without c_i, so whichever functional the loop minimises is the one differenced.
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
    // Broadcast raw coordinates, not vectors: unpacked ONCE per round on the driver, so the
    // per-row scan runs the array kernel with distanceUpTo's shrinking-bound early exit instead
    // of dereferencing a wrapper and re-dispatching on the vector's type at every comparison —
    // same reasoning as LloydKMeans's assignment broadcast.
    val broadcastCentroids = sc.broadcast(centroids.map(_.toArray))
    val dist = distance
    val geom = geometry // local val: the UDF must not capture the enclosing clusterer

    // ONE ordered fold instead of struct-select + groupBy + three sums. Same shape as Lloyd's
    // (measured 5.3x there, 5.09.2026): the per-row work is an opaque distance loop either way,
    // so Catalyst has nothing to optimise and the DataFrame form only adds a VectorUDT round-trip
    // per row plus a shuffle to group k keys. Accumulator is centroid-major, three slots each
    // (mass, error, utility) — k*3 doubles, so ORDERED merging costs nothing and buys back the
    // reproducibility that task-completion order would spend. It matters more here than in Lloyd:
    // these sums decide WHICH centroid breathing deletes, so a last-bit difference is not a
    // rounding detail, it can change the tree of decisions.
    val acc = PartitionAggregator.aggregateDoublesOrdered(Weights.toRdd(points), centroids.length * 3) {
      (arr, row) =>
        val coords = row._1.toArray
        val w      = row._2
        val cs     = broadcastCentroids.value

        var nearestIndex = 0
        var nearestDistance = Double.MaxValue
        var secondNearestDistance = Double.MaxValue
        var i = 0
        while (i < cs.length) {
          // Bound frozen at Double.MaxValue, so this NEVER early-exits — deliberately: shrinking
          // the bound against d2 was measured 5.09.2026 to REGRESS a top-2 scan at every k >= 32
          // (see CentroidIteration.PartialAssign on the Flink side for the same finding; d2 is
          // nothing like a small fixed ε). The frozen bound still buys the sqrt-free ordinal path
          // for Euclidean (see DistanceMetric.distanceUpToOrdinal) without touching that decision.
          val d = dist.distanceUpToOrdinal(coords, cs(i), Double.MaxValue)
          if (d < nearestDistance) {
            secondNearestDistance = nearestDistance
            nearestDistance = d
            nearestIndex = i
          } else if (d < secondNearestDistance) {
            secondNearestDistance = d
          }
          i += 1
        }
        val errorCost = geom.pointCostFromOrdinal(nearestDistance)
        val secondCost =
          if (secondNearestDistance == Double.MaxValue) errorCost
          else geom.pointCostFromOrdinal(secondNearestDistance)

        val base = nearestIndex * 3
        arr(base)     += w                              // mass
        arr(base + 1) += w * errorCost                  // error   = Σ w·φ(d1)
        arr(base + 2) += w * (secondCost - errorCost)   // utility = Σ w·(φ(d2) − φ(d1))
    }

    broadcastCentroids.unpersist(blocking = false)

    Array.tabulate(centroids.length) { i =>
      val base = i * 3
      BreathingKMeans.Stats(mass = acc(base), error = acc(base + 1), utility = acc(base + 2))
    }
  }
}

private object BreathingKMeans {

  /** Per-centroid aggregates: `mass` = Σw, `error` = φ(c) = Σ w·φ(d1),
   *  `utility` = U(c) = Σ w·(φ(d2) − φ(d1)), with φ the geometry's point cost. */
  private final case class Stats(mass: Double, error: Double, utility: Double)
}

