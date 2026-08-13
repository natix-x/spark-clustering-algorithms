package clustering.algorithms.kmedoids.hybrid

import clustering.algorithms.kmedoids.KMedoidsModel
import clustering.algorithms.kmedoids.components.MedoidRefinement
import clustering.core.{Clusterer, Weights}
import clustering.distance.{DistanceMetric, EuclideanDistance}
import org.apache.spark.sql.DataFrame
import org.log4s.getLogger

/** PAMAE (Song, Jin, Kim & Lee, **KDD 2017**) — the established parallel k-medoids algorithm, and
 *  the only medoid entry whose optimisation sees the entire data.
 *
 *  Phase I is [[CLARA]] (global search over samples, winner picked by cost on the entire data);
 *  phase II is `refineIters` Voronoi updates over the entire data ([[MedoidRefinement]]). So
 *  `pamae` and `clara` differ by exactly one phase — a controlled experiment, not two unrelated
 *  implementations. See `docs/kmedoids_docs.md`.
 *
 *  @param inner       driver-local solver used inside phase I (`pam` | `fastpam` | `fasterpam`)
 *  @param refineIters phase-II iterations; the paper uses 1
 *  @param poolSize    phase-II candidate pool — the accuracy-vs-cost knob
 */
class PAMAE(
  val k: Int,
  val numSamples: Int = 5,
  val sampleSize: Int = 1000,
  val maxIter: Int = 100,
  val refineIters: Int = 1,
  val poolSize: Int = 2000,
  val inner: String = "pam",
  val distance: DistanceMetric = EuclideanDistance,
  val seed: Long = 42L
) extends Clusterer {

  require(refineIters >= 1, s"refineIters must be >= 1, got $refineIters")

  private val logger = getLogger

  override def fit(data: DataFrame): KMedoidsModel = {
    val points = Weights.withWeights(data)

    // Phase I — parallel seeding; its full-data cost comes back from CLARA's batched evaluation.
    val seeding = new CLARA(k, numSamples, sampleSize, maxIter, distance, inner, seed).seedMedoids(data)

    // Phase II — parallel refinement over the entire data.
    val candidatePool = MedoidRefinement.sampleCandidatePool(points, seeding.rowCount, poolSize, seed)
    val refined = MedoidRefinement.refine(
      Weights.toRdd(points), seeding.model.medoids, candidatePool, distance, refineIters)

    // What the "entire data" half bought is the paper's headline number, so every run records it.
    logger.info(f"pamae: seedingCost=${seeding.cost}%.4f refinedCost=${refined.cost}%.4f " +
      f"gain=${(seeding.cost - refined.cost) / math.max(seeding.cost, 1e-12) * 100}%.2f%% " +
      s"iterations=${refined.iterations} poolSize=${candidatePool.length} inner=$inner")

    new KMedoidsModel(refined.medoids, distance)
  }
}
