package clustering.algorithms.kmedoids.components

import clustering.algorithms.kmedoids.hybrid.PAMAE
import clustering.core.{Columns, NearestPrototypeModel}
import clustering.distance.DistanceMetric
import clustering.utils.PartitionAggregator
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col

/** The distributed medoid update — one Voronoi ("k-means-like") iteration over the **entire**
 *  dataset, and the only new distributed primitive the medoid family needs.
 *
 *  Phase II of [[PAMAE]] (Song et al., KDD 2017). Pattern P1: one broadcast, one map, one array
 *  aggregation per iteration, no shuffle.
 *
 *  Candidate pool, monotonicity, weights and why the aggregation is an RDD fold rather than
 *  Catalyst: `docs/kmedoids_docs.md`.
 */
private[kmedoids] object MedoidRefinement {

  /** @param medoids    the refined medoids
   *  @param cost       Σ_x w_x·d(x, medoid(x)) after the last accepted iteration
   *  @param iterations iterations actually executed (stops early when nothing improves) */
  final case class Result(medoids: Array[Vector], cost: Double, iterations: Int)

  def refine(
    points: RDD[(Vector, Double)],
    initialMedoids: Array[Vector],
    candidatePool: Array[Vector],
    distance:DistanceMetric,
    maxIterations: Int
  ): Result = {
    val sc = points.sparkContext
    val slotCount = initialMedoids.length

    var medoids = initialMedoids
    var cost = Double.MaxValue
    var iteration = 0
    var improved  = true

    while (improved && iteration < maxIterations) {
      // Incumbents are always candidates — that is what makes the objective non-increasing.
      val candidates = candidatePool ++ medoids
      val slotMajor = groupCandidatesBySlot(candidates, medoids, distance)
      val slotCandidates = slotMajor.candidates
      val slotBounds = slotMajor.bounds

      // One distributed pass, raw arrays unpacked once: slot pick is a pure argmin (`nearestRaw`,
      // sqrt-free), but the cost is a running SUM of real distances — squaring would reorder
      // candidates — so it keeps raw-array `compute`, just without the Vector-overload dispatch.
      val broadcastCandidates = sc.broadcast(slotCandidates.map(_.toArray))
      val broadcastMedoids = sc.broadcast(medoids.map(_.toArray))
      val broadcastBounds = sc.broadcast(slotBounds)
      val metric = distance
      val candidateCosts = PartitionAggregator.aggregateDoubles(points, slotCandidates.length) {
        (costs, pointAndWeight) =>
          val (point, weight) = pointAndWeight
          val coords = point.toArray
          val slot = NearestPrototypeModel.nearestRaw(coords, broadcastMedoids.value, metric)
          val candidatesOfSlot = broadcastCandidates.value
          var index = broadcastBounds.value(slot)
          val end   = broadcastBounds.value(slot + 1)
          while (index < end) {
            costs(index) += weight * metric.compute(coords, candidatesOfSlot(index))
            index += 1
          }
      }
      broadcastCandidates.destroy(); broadcastMedoids.destroy(); broadcastBounds.destroy()

      val nextMedoids = new Array[Vector](slotCount)
      var nextCost = 0.0
      var slot = 0
      while (slot < slotCount) {
        val best = bestCandidateOfSlot(slot, slotCandidates, slotBounds, candidateCosts, medoids)
        nextMedoids(slot) = best._1
        nextCost += best._2      // an empty slot contributes 0 and keeps its medoid
        slot += 1
      }

      // `nextCost` is the objective of `nextMedoids` under the CURRENT assignment; reassigning in
      // the next iteration can only lower it, so the sequence is non-increasing.
      improved = !nextMedoids.sameElements(medoids) && nextCost < cost
      cost = math.min(cost, nextCost)
      if (improved) medoids = nextMedoids
      iteration += 1
    }

    Result(medoids, cost, iteration)
  }

  /** Candidates flattened in slot order, plus the slot boundaries into that flat array — so the
   *  distributed pass can index a slot's candidates without a shuffle. */
  private final case class SlotMajorCandidates(candidates: Array[Vector], bounds: Array[Int])

  private def groupCandidatesBySlot(
    candidates: Array[Vector],
    medoids: Array[Vector],
    distance: DistanceMetric
  ): SlotMajorCandidates = {
    val medoidsRaw = medoids.map(_.toArray) // unpacked once, reused by every candidate below
    val perSlot = Array.fill(medoids.length)(List.newBuilder[Vector])
    var index = 0
    while (index < candidates.length) {
      perSlot(NearestPrototypeModel.nearestRaw(candidates(index).toArray, medoidsRaw, distance)) += candidates(index)
      index += 1
    }
    val groups = perSlot.map(_.result().toArray)
    SlotMajorCandidates(groups.flatten, groups.scanLeft(0)(_ + _.length))
  }

  /** Keeps the incumbent unless a candidate is STRICTLY better, so ties never move a medoid. The
   *  incumbent is normally in its own slot (distance 0 to itself) because it was appended to the
   *  pool; the search starts from it. */
  private def bestCandidateOfSlot(
    slot: Int,
    slotCandidates: Array[Vector],
    slotBounds: Array[Int],
    candidateCosts: Array[Double],
    medoids: Array[Vector]
  ): (Vector, Double) = {
    val from = slotBounds(slot)
    val until = slotBounds(slot + 1)

    var bestIndex = -1
    var bestCost = 0.0
    var index = from
    while (index < until) {
      if (slotCandidates(index) == medoids(slot)) { bestIndex = index; bestCost = candidateCosts(index) }
      index += 1
    }
    index = from
    while (index < until) {
      if (bestIndex < 0 || candidateCosts(index) < bestCost) {
        bestIndex = index
        bestCost = candidateCosts(index)
      }
      index += 1
    }

    if (bestIndex >= 0) (slotCandidates(bestIndex), bestCost) else (medoids(slot), 0.0)
  }

  /** Uniformly sampled candidate pool, collected once and reused by every iteration. Sampling is
   *  uniform over ROWS, not weight — candidates only need to cover the space.
   *
   *  Uniform over the WHOLE dataset, which is the point: this pool is the set the refinement is
   *  allowed to choose representatives from, so a pool drawn from one region of an ordered input
   *  would quietly restrict the phase that exists precisely to see all of the data. Hence the
   *  over-draw and the shuffle in [[DriverSample]] rather than a `take` on the sampled frame. */
  def sampleCandidatePool(data: DataFrame, rowCount: Long, poolSize: Int, seed: Long): Array[Vector] = {
    val features = data.select(col(Columns.Features))
    val drawn =
      if (poolSize >= rowCount) features.collect()
      else features
        .sample(withReplacement = false,
                fraction = DriverSample.inclusionProbability(poolSize, rowCount),
                seed = seed)
        .collect()
    DriverSample.takeRandom(drawn, poolSize, seed).map(_.getAs[Vector](Columns.Features))
  }
}
