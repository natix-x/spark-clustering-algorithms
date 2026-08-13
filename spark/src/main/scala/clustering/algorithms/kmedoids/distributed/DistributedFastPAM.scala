package clustering.algorithms.kmedoids.distributed

import clustering.algorithms.kmedoids.KMedoidsModel
import clustering.algorithms.kmedoids.components.SwapMove
import clustering.core.{Clusterer, Weights}
import clustering.distance.{DistanceMetric, EuclideanDistance}
import clustering.utils.PartitionAggregator
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame

/** FastPAM1 with the Σ-over-points spread across the CLUSTER — the O(n²) distance matrix is NEVER
 *  materialised, the pairwise distances it would hold are recomputed inside the partitions.
 *
 *  Candidate coordinates (all n points, O(n·d)) are collected once and broadcast; the points stay
 *  in the partitioned RDD; each partition folds its local points into per-candidate partial sums;
 *  the driver merges the partials and picks the move. One job per BUILD step (k of them) and one
 *  per SWAP round (≤ `maxIter`).
 *
 */
class DistributedFastPAM(
  val k: Int,
  val maxIter:  Int = 100,
  val distance: DistanceMetric = EuclideanDistance
) extends Clusterer {

  override def fit(data: DataFrame): KMedoidsModel = {
    val points = Weights.toRdd(data).cache()

    val candidates = points.map(_._1).collect()
    require(candidates.length >= k,
      s"Dataset too small: n=${candidates.length} points but k=$k medoids requested.")
    val broadcastCandidates = points.sparkContext.broadcast(candidates)

    val medoids  = buildPhase(points, broadcastCandidates)
    val isMedoid = new Array[Boolean](candidates.length)
    medoids.foreach(index => isMedoid(index) = true)

    var iteration = 0
    var improved = true
    while (improved && iteration < maxIter) {
      improved = applyBestSwap(points, broadcastCandidates, medoids, isMedoid)
      iteration += 1
    }

    broadcastCandidates.destroy()
    points.unpersist(blocking = false)
    new KMedoidsModel(medoids.map(candidates), distance)
  }

  /** Greedy BUILD, one distributed job per medoid. */
  private def buildPhase(
    points:              RDD[(Vector, Double)],
    broadcastCandidates: Broadcast[Array[Vector]]
  ): Array[Int] = {
    val candidateCount = broadcastCandidates.value.length
    val medoids        = new Array[Int](k)
    val isSelected     = new Array[Boolean](candidateCount)
    val metric         = distance

    // First medoid: minimise Σ_j w_j·d(candidate, j).
    val totalDistances = PartitionAggregator.aggregateDoubles(points, candidateCount) {
      (totals, pointAndWeight) =>
        val (point, weight) = pointAndWeight
        val candidates      = broadcastCandidates.value
        var candidate = 0
        while (candidate < candidateCount) {
          totals(candidate) += weight * metric.compute(candidates(candidate), point)
          candidate += 1
        }
    }
    medoids(0) = indexOfBest(totalDistances, isSelected, preferSmaller = true)
    isSelected(medoids(0)) = true

    // Remaining k-1: maximise Σ_j w_j·max(0, d(j, nearest selected) − d(candidate, j)).
    var selectedCount = 1
    while (selectedCount < k) {
      val broadcastSelected =
        points.sparkContext.broadcast(medoids.take(selectedCount).map(broadcastCandidates.value))
      val gains = PartitionAggregator.aggregateDoubles(points, candidateCount) {
        (totals, pointAndWeight) =>
          val (point, weight) = pointAndWeight
          val candidates      = broadcastCandidates.value
          val selected        = broadcastSelected.value
          var distanceToNearestMedoid = Double.MaxValue
          var slot = 0
          while (slot < selected.length) {
            val d = metric.compute(selected(slot), point)
            if (d < distanceToNearestMedoid) distanceToNearestMedoid = d
            slot += 1
          }
          var candidate = 0
          while (candidate < candidateCount) {
            val reduction = distanceToNearestMedoid - metric.compute(candidates(candidate), point)
            if (reduction > 0.0) totals(candidate) += weight * reduction
            candidate += 1
          }
      }
      broadcastSelected.destroy()
      medoids(selectedCount) = indexOfBest(gains, isSelected, preferSmaller = false)
      isSelected(medoids(selectedCount)) = true
      selectedCount += 1
    }
    medoids
  }

  /** One distributed FastPAM1 SWAP round. Mutates `medoids`/`isMedoid` in place when an improving
   *  swap is found; returns whether one was applied. */
  private def applyBestSwap(
    points:              RDD[(Vector, Double)],
    broadcastCandidates: Broadcast[Array[Vector]],
    medoids:             Array[Int],
    isMedoid:            Array[Boolean]
  ): Boolean = {
    val candidateCount   = broadcastCandidates.value.length
    val slotCount        = k
    val metric           = distance
    val broadcastMedoids = points.sparkContext.broadcast(medoids.map(broadcastCandidates.value))

    // terms(candidate)                                = shared(candidate)
    // terms(candidateCount + candidate*k + slot)      = removeLoss(candidate)(slot)
    val terms = PartitionAggregator.aggregateDoubles(points, candidateCount + candidateCount * slotCount) {
      (accumulator, pointAndWeight) =>
        val (point, weight) = pointAndWeight
        val candidates      = broadcastCandidates.value
        val medoidPoints    = broadcastMedoids.value

        var nearest       = Double.MaxValue
        var secondNearest = Double.MaxValue
        var nearestSlot   = -1
        var slot = 0
        while (slot < slotCount) {
          val d = metric.compute(medoidPoints(slot), point)
          if (d < nearest) { secondNearest = nearest; nearest = d; nearestSlot = slot }
          else if (d < secondNearest) secondNearest = d
          slot += 1
        }

        var candidate = 0
        while (candidate < candidateCount) {
          val distanceToCandidate = metric.compute(candidates(candidate), point)
          val sharedContribution =
            if (distanceToCandidate < nearest) weight * (distanceToCandidate - nearest) else 0.0
          accumulator(candidate) += sharedContribution
          val costIfOwnSlotRemoved = weight * (math.min(secondNearest, distanceToCandidate) - nearest)
          accumulator(candidateCount + candidate * slotCount + nearestSlot) +=
            costIfOwnSlotRemoved - sharedContribution
          candidate += 1
        }
    }
    broadcastMedoids.destroy()

    var best = SwapMove.None
    var candidate = 0
    while (candidate < candidateCount) {
      if (!isMedoid(candidate)) {
        val shared = terms(candidate)
        val base = candidateCount + candidate * slotCount
        var slot = 0
        while (slot < slotCount) {
          val delta = shared + terms(base + slot)
          if (delta < 0.0) best = SwapMove.preferred(best, SwapMove(delta, slot, candidate))
          slot += 1
        }
      }
      candidate += 1
    }

    if (SwapMove.isImprovement(best)) {
      isMedoid(medoids(best.slot)) = false
      medoids(best.slot) = best.candidate
      isMedoid(best.candidate) = true
      true
    } else {
      false
    }
  }

  /** Index of the largest (or smallest) still-unselected entry; ties go to the lowest index. */
  private def indexOfBest(values: Array[Double], isSelected: Array[Boolean], preferSmaller: Boolean): Int = {
    var best = -1
    var bestValue = if (preferSmaller) Double.MaxValue else Double.NegativeInfinity
    var index = 0
    while (index < values.length) {
      val better = if (preferSmaller) values(index) < bestValue else values(index) > bestValue
      if (!isSelected(index) && better) { bestValue = values(index); best = index }
      index += 1
    }
    best
  }
}
