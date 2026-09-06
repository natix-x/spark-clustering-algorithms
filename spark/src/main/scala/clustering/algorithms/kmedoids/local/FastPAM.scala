package clustering.algorithms.kmedoids.local

import clustering.algorithms.kmedoids.components._
import clustering.algorithms.kmedoids.KMedoidsModel
import clustering.distance.{DistanceMetric, EuclideanDistance}
import org.apache.spark.ml.linalg.Vector

/** FastPAM1 (Schubert & Rousseeuw 2019) — the EXACT rung of the ladder.
 *
 *  It performs the same search as the original PAM (Kaufman & Rousseeuw 1990): one best swap per
 *  iteration, over all (slot, candidate) pairs. The O(k) gain is purely in how a candidate is
 *  scored — ONE O(n) pass yields its Δ for ALL k slots at once ([[SwapDeltas]]), instead of an
 *  evaluation per pair — so the paper presents it as a runtime optimisation that returns the
 *  IDENTICAL medoids, not an approximation. That is why no separate exhaustive `pam` entry
 *  exists here: it would produce the same answer for a further factor k.
 *
 *  Driver-local; the candidate scan runs on the driver's cores.
 *  See `docs/kmedoids_docs.md`.
 */
class FastPAM(
  val k: Int,
  val maxIter: Int = 100,
  val distance: DistanceMetric = EuclideanDistance
) extends DriverLocalKMedoids {

  override def fitLocal(points: Array[Vector], weights: Array[Double]): KMedoidsModel = {
    val pointCount = points.length
    require(pointCount >= k, s"Dataset too small: n=$pointCount points but k=$k medoids requested.")
    require(weights.length == pointCount, s"weights (${weights.length}) must match points ($pointCount)")

    val distances = DistanceMatrix.pairwise(points, distance)
    var medoids = MedoidBuildPhase.selectInitialMedoids(distances, k, weights)
    val cache= new NearestMedoidCache(pointCount)

    var iteration = 0
    var improved = true
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

    // One O(n) Δ pass per candidate, ascending, so `preferred` sees candidates in index order and
    // the tie rule resolves to the lowest index.
    val deltasBySlot = new Array[Double](k)   // reused across candidates: the scan allocates nothing
    var best = SwapMove.None
    var candidate = 0
    while (candidate < distances.pointCount) {
      if (!isMedoid(candidate)) {
        best = SwapMove.preferred(best,
          SwapDeltas.bestMoveForCandidate(distances, candidate, weights, cache, deltasBySlot))
      }
      candidate += 1
    }
    best
  }
}
