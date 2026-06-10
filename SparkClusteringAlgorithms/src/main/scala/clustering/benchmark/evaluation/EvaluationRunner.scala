package clustering.benchmark.evaluation

import clustering.benchmark.config.EvaluationSpec
import clustering.core.Model
import clustering.distance.DistanceMetric
import clustering.evaluation.SilhouetteEvaluator
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col
import org.apache.spark.storage.StorageLevel

/** Quality metrics computed after `fit()`.
 *
 *  Bundles silhouette (optional, gated by `metrics` config) with cheap
 *  per-cluster statistics. Silhouette is O(n^2) in
 *  [[clustering.evaluation.SilhouetteEvaluator]], so for big data sets the
 *  caller should set `evaluation.sampleSize` to a tractable number
 *  (e.g. 10_000).
 */
final class EvaluationRunner(distance: DistanceMetric) {

  /** Runs the evaluation suite over the dataset.
   *  Optionally accepts a pre-computed row count to avoid a redundant
   *  `data.count()` scan inside silhouette sampling.
   */
  def run(model: Model, data: DataFrame, spec: EvaluationSpec, knownTotalRows: Option[Long] = None): EvaluationResult = {
    val labeled = model.labeledData(data)
    labeled.persist(StorageLevel.MEMORY_AND_DISK)

    val sizes = labeled
      .groupBy(col("prediction"))
      .count()
      .collect()
      .map(r => r.getInt(0) -> r.getLong(1))
      .toMap

    val totalLabeled  = sizes.values.sum
    val noiseCount    = sizes.getOrElse(-1, 0L)
    val noiseFraction = if (totalLabeled == 0L) 0.0 else noiseCount.toDouble / totalLabeled
    val nClusters     = sizes.keys.count(_ >= 0)

    val silhouette: Option[Double] =
      if (!spec.metrics.contains("silhouette")) None
      else Some(computeSilhouette(model, data, spec, knownTotalRows))

    labeled.unpersist(blocking = false)

    EvaluationResult(
      silhouette    = silhouette,
      nClusters     = nClusters,
      noiseFraction = noiseFraction,
      clusterSizes  = sizes
    )
  }

  /** Silhouette on the full data set or a uniform sample, depending on spec.
   *  Sampling is essential for big data — the underlying evaluator is O(n^2). */
  private def computeSilhouette(model: Model, data: DataFrame, spec: EvaluationSpec, knownTotalRows: Option[Long]): Double = {
    val evaluator = new SilhouetteEvaluator(distance)
    spec.sampleSize.filter(_ > 0) match {
      case None =>
        evaluator.evaluate(model, data)
      case Some(s) =>
        val total = knownTotalRows.getOrElse(data.count())
        if (total <= s) evaluator.evaluate(model, data)
        else {
          val fraction = math.min(1.0, s.toDouble / total)
          val sampled  = data.sample(withReplacement = false, fraction = fraction, seed = spec.seed)
          evaluator.evaluate(model, sampled)
        }
    }
  }
}

final case class EvaluationResult(
  silhouette:    Option[Double],
  nClusters:     Int,
  noiseFraction: Double,
  clusterSizes:  Map[Int, Long]
)