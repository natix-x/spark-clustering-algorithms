package clustering.algorithms.coreset

import clustering.algorithms.coreset.components.LightweightCoreset
import clustering.algorithms.coreset.inner.InnerSolver
import clustering.core.{Clusterer, Weights}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.DataFrame
import org.apache.spark.storage.StorageLevel
import org.log4s.getLogger

/** Coreset clustering — data reduction first, then any inner algorithm on the reduction.
 *
 *  `fit` is two things in sequence, and they are deliberately independent: the lightweight coreset
 *  of Bachem, Lucic & Krause (KDD 2018) is drawn from the full data in three distributed passes
 *  ([[LightweightCoreset]]), and an [[InnerSolver]] then clusters those `m` weighted rows
 *  driver-locally. Everything after the draw is O(m), so the reduction is the only part whose cost
 *  scales with the dataset — which is what makes `m` the benchmark's accuracy-vs-cost knob in its
 *  purest form: it moves the boundary between distributed and local work directly.
 *
 *  ==What the entry measures==
 *  Three inner algorithms compose with one reduction (`kmeans`, `medoids`, `hac`), so a sweep over
 *  `m` reports quality against network/driver cost for three paradigms at once, and `coreset +
 *  medoids` completes the thesis' three-way comparison of principled reductions of the medoid
 *  objective (uniform sampling in `clara`, one-batch estimation in `onebatchpam`,
 *  importance-weighted reduction with an error bound here).
 *
 *  ==The caller still owns the cache==
 *  The construction reads the input three times and persists nothing. Only a prepared frame this
 *  fit CREATED — the spherical geometry's normalised copy — is cached and released here, exactly as
 *  the k-means entries do.
 */
class Coreset(
  val k: Int,
  val m: Int,
  val solver: InnerSolver,
  val seed: Long = 42L
) extends Clusterer {

  require(k >= 1, s"k must be >= 1, got $k")
  require(m >= k, s"coreset size m=$m must be at least k=$k")

  private val logger = getLogger

  override def fit(data: DataFrame): CoresetModel = {
    logger.info(s"coreset: fit start k=$k m=$m inner=${solver.name} seed=$seed " +
      s"fitDistance=${solver.fitDistance.getClass.getSimpleName}")
    val startNanos = System.nanoTime()

    val projected = solver.prepare(Weights.withWeights(data))
    // Same rule as LloydKMeans.initialize: cache only a frame this fit computed. A no-op
    // `prepare` is a bare projection of the caller's cached data, and persisting it would store a
    // second copy of `features`.
    val prepared = if (solver.transformsFeatures) projected.persist(StorageLevel.MEMORY_AND_DISK) else projected

    val coreset =
      try LightweightCoreset.build(prepared, m, solver.fitDistance, seed)
      finally if (solver.transformsFeatures) prepared.unpersist(blocking = false)

    require(coreset.size >= k,
      s"coreset drew ${coreset.size} rows but k=$k — raise m (requested m=$m, the draw is " +
        "Poisson so its size is only m in expectation)")

    val solution = solver.solve(coreset.points, coreset.weights)

    val durationMs = (System.nanoTime() - startNanos) / 1000000L
    logger.info(f"coreset: fit done k=$k m=$m drawn=${coreset.size} inner=${solver.name} " +
      f"mass=${coreset.mass}%.4f coresetMass=${coreset.coresetMass}%.4f " +
      s"iterationsRun=${solution.iterationsRun} durationMs=$durationMs")

    new CoresetModel(
      solution.prototypes.map(Vectors.dense),
      solver.modelDistance,
      solution.iterationsRun,
      coresetSize = coreset.size,
      requestedSize = m,
      inner = solver.name
    )
  }
}
