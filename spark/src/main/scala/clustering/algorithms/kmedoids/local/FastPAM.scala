package clustering.algorithms.kmedoids.local

import clustering.algorithms.kmedoids.components._
import clustering.algorithms.kmedoids.KMedoidsModel
import clustering.distance.{DistanceMetric, EuclideanDistance}
import clustering.utils.DriverParallelism
import org.apache.spark.ml.linalg.Vector

/** FastPAM1 (Schubert & Rousseeuw 2019) — same search as [[PAM]] (one best swap per iteration),
 *  but a candidate's Δ for ALL k slots comes from ONE O(n) pass over the points
 *  ([[SwapDeltas]]) instead of a configuration-cost recomputation per (slot, candidate) pair.
 *
 *  Driver-local like [[PAM]]; the candidate scan runs on the driver's cores.
 *  See `docs/kmedoids_docs.md`.
 */
class FastPAM(
  val k:        Int,
  val maxIter:  Int            = 100,
  val distance: DistanceMetric = EuclideanDistance
) extends DriverLocalKMedoids {

  override def fitLocal(points: Array[Vector], weights: Array[Double]): KMedoidsModel = {
    val pointCount = points.length
    require(pointCount >= k, s"Dataset too small: n=$pointCount points but k=$k medoids requested.")
    require(weights.length == pointCount, s"weights (${weights.length}) must match points ($pointCount)")

    val distances = DistanceMatrix.pairwise(points, distance)
    var medoids   = MedoidBuildPhase.selectInitialMedoids(distances, k, weights)
    val cache     = new NearestMedoidCache(pointCount)

    var iteration = 0
    var improved  = true
    while (improved && iteration < maxIter) {
      cache.refresh(distances, medoids)
      val move = bestSwap(distances, medoids, weights, cache)
      improved = SwapMove.isImprovement(move)
      if (improved) {
        medoids = medoids.clone()
        medoids(move.slot) = move.candidate
      }
      iteration += 1
    }

    new KMedoidsModel(medoids.map(points), distance)
  }

  /** Best (slot, candidate) swap over all non-medoid candidates. */
  private def bestSwap(
    distances: DistanceMatrix,
    medoids:   Array[Int],
    weights:   Array[Double],
    cache:     NearestMedoidCache
  ): SwapMove = {
    val isMedoid = new Array[Boolean](distances.pointCount)
    medoids.foreach(m => isMedoid(m) = true)

    // One O(n) Δ pass per candidate.
    DriverParallelism
      .sliceRangesForWork(distances.pointCount, distances.pointCount.toLong)
      .par.map { case (from, until) =>
      val deltasBySlot = new Array[Double](k)   // one scratch buffer per slice, never shared
      var best         = SwapMove.None
      var candidate    = from
      while (candidate < until) {
        if (!isMedoid(candidate)) {
          best = SwapMove.preferred(best,
            SwapDeltas.bestMoveForCandidate(distances, candidate, weights, cache, deltasBySlot))
        }
        candidate += 1
      }
      best
    }.seq.reduce(SwapMove.preferred)
  }
}
