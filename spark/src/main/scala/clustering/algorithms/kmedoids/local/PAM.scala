package clustering.algorithms.kmedoids

import clustering.distance.{DistanceMetric, EuclideanDistance}
import clustering.utils.DriverParallelism
import org.apache.spark.ml.linalg.Vector

/** Partitioning Around Medoids (Kaufman & Rousseeuw 1990) — the exhaustive rung: every one of the
 *  k·(n−k) swaps is scored by recomputing the full configuration cost, and the single best swap is
 *  applied per iteration.
 *
 *  Driver-local: O(n²) matrix in driver memory, O(k²·n) per iteration, so n ≲ 10 000. Use `clara`
 *  or `distfastpam` beyond that. Rationale, parallelism and tie rules: `docs/kmedoids_docs.md`.
 */
class PAM(
  val k: Int,
  val maxIter:  Int = 100,
  val distanceMetrics: DistanceMetric = EuclideanDistance
) extends DriverLocalKMedoids {

  override def fitLocal(points: Array[Vector], weights: Array[Double]): KMedoidsModel = {
    val pointCount = points.length
    require(pointCount >= k, s"Dataset too small: n=$pointCount points but k=$k medoids requested.")
    require(weights.length == pointCount, s"weights (${weights.length}) must match points ($pointCount)")

    val distances = DistanceMatrix.pairwise(points, distanceMetrics)
    var medoids   = MedoidBuildPhase.selectInitialMedoids(distances, k, weights)

    var iteration = 0
    var improved  = true
    while (improved && iteration < maxIter) {
      val move = bestExhaustiveSwap(distances, medoids, weights)
      improved = SwapMove.isImprovement(move)
      if (improved) {
        medoids = medoids.clone()
        medoids(move.slot) = move.candidate
      }
      iteration += 1
    }

    new KMedoidsModel(medoids.map(points), distanceMetrics)
  }

  /** Scores every (slot, candidate) pair by the cost of the resulting configuration — the 1990
   *  formulation, kept naive on purpose. Parallel over candidates. */
  private def bestExhaustiveSwap(
    distances: DistanceMatrix,
    medoids:   Array[Int],
    weights:   Array[Double]
  ): SwapMove = {
    val pointCount  = distances.pointCount
    val currentCost = configurationCost(distances, medoids, weights)
    val isMedoid    = new Array[Boolean](pointCount)
    medoids.foreach(m => isMedoid(m) = true)

    // Each candidate costs k configuration-cost evaluations of O(n·k).
    val workPerCandidate = pointCount.toLong * k * k
    DriverParallelism.sliceRangesForWork(pointCount, workPerCandidate).par.map { case (from, until) =>
      val trialMedoids = medoids.clone()
      var best         = SwapMove.None
      var candidate    = from
      while (candidate < until) {
        if (!isMedoid(candidate)) {
          var slot = 0
          while (slot < k) {
            val replaced = trialMedoids(slot)
            trialMedoids(slot) = candidate
            val delta = configurationCost(distances, trialMedoids, weights) - currentCost
            trialMedoids(slot) = replaced
            if (delta < 0.0) best = SwapMove.preferred(best, SwapMove(delta, slot, candidate))
            slot += 1
          }
        }
        candidate += 1
      }
      best
    }.seq.reduce(SwapMove.preferred)
  }

  /** Σ_j w_j · min over medoids d(j, medoid). */
  private def configurationCost(
    distances: DistanceMatrix,
    medoids:   Array[Int],
    weights:   Array[Double]
  ): Double = {
    var total = 0.0
    var point = 0
    while (point < distances.pointCount) {
      var nearest = Double.MaxValue
      var slot    = 0
      while (slot < medoids.length) {
        val d = distances(point, medoids(slot))
        if (d < nearest) nearest = d
        slot += 1
      }
      total += weights(point) * nearest
      point += 1
    }
    total
  }
}
