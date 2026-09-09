package clustering.algorithms.kmedoids.local

import clustering.algorithms.kmedoids.components._
import clustering.algorithms.kmedoids.KMedoidsModel
import clustering.distance.{DistanceMetric, EuclideanDistance}
import org.apache.spark.ml.linalg.Vector

import scala.util.Random

/** FasterPAM (Schubert & Rousseeuw, *Information Systems* 2021) — the same FastPAM1 Δ as
 *  [[FastPAM]], but a candidate whose best Δ < 0 is swapped in **immediately** and the caches are
 *  refreshed before the next candidate is looked at. Many swaps per pass instead of one.
 *
 *  Eager swapping is order-dependent, so candidates are visited in a `seed`-derived shuffled
 *  order; same seed and same n ⇒ same medoids. The eager loop is inherently sequential — only the
 *  distance matrix and the cache refreshes use the driver's cores.
 *
 *  Deviations from the paper (BUILD instead of LAB, caches recomputed rather than updated) and
 *  what they buy the measurement: `docs/kmedoids_docs.md`.
 *
 *  @param maxIter maximum full passes over the candidate set; a pass with no swap ends the search
 */
class FasterPAM(
  val k:        Int,
  val maxIter:  Int            = 100,
  val distance: DistanceMetric = EuclideanDistance,
  val seed:     Long           = 42L
) extends DriverLocalKMedoids {

  override def fitLocal(points: Array[Vector], weights: Array[Double]): KMedoidsModel = {
    val pointCount = points.length
    require(pointCount >= k, s"Dataset too small: n=$pointCount points but k=$k medoids requested.")
    require(weights.length == pointCount, s"weights (${weights.length}) must match points ($pointCount)")

    val distances = DistanceMatrix.pairwise(points, distance)
    val medoids   = MedoidBuildPhase.selectInitialMedoids(distances, k, weights)
    val cache     = new NearestMedoidCache(pointCount)
    cache.refresh(distances, medoids)

    val isMedoid = new Array[Boolean](pointCount)
    medoids.foreach(m => isMedoid(m) = true)

    val visitOrder    = new Random(seed).shuffle((0 until pointCount).toVector).toArray
    val deltasBySlot  = new Array[Double](k)

    var pass     = 0
    var improved = true
    while (improved && pass < maxIter) {
      improved = false
      var visited = 0
      while (visited < pointCount) {
        val candidate = visitOrder(visited)
        if (!isMedoid(candidate)) {
          val move = SwapDeltas.bestMoveForCandidate(distances, candidate, weights, cache, deltasBySlot)
          if (SwapMove.isImprovement(move)) {
            isMedoid(medoids(move.slot)) = false
            medoids(move.slot)           = candidate
            isMedoid(candidate)          = true
            // Incremental update, not a full O(n·k) refresh: only slot move.slot moved, so almost
            // every point's top-2 is untouched. See NearestMedoidCache.updateAfterSwap.
            cache.updateAfterSwap(distances, medoids, move.slot)
            improved = true
          }
        }
        visited += 1
      }
      pass += 1
    }

    new KMedoidsModel(medoids.map(points), distance)
  }
}
