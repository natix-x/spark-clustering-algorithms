package clustering.benchmark.evaluation

import clustering.benchmark.config.EvaluationSpec
import clustering.core.{Columns, Model}
import clustering.distance.DistanceMetric
import clustering.evaluation.SilhouetteEvaluator
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.IntegerType

final class EvaluationRunner(distance: DistanceMetric) {

  def run(model: Model, data: DataFrame, spec: EvaluationSpec, knownTotalRows: Option[Long] = None): EvaluationResult = {
    val metrics       = spec.metrics.toSet
    val wantSizes     = metrics.contains("clusterSizes")
    val wantNoise     = metrics.contains("noiseFraction")
    val wantNClusters = metrics.contains("nClusters")

    // clusterSizes, noiseFraction and nClusters all derive from one assignment
    // scan; run it once iff at least one of them is requested. Everything the
    // config didn't ask for stays None, so the result omits it (see RunResult).
    val sizes: Option[Map[Int, Long]] =
      if (wantSizes || wantNoise || wantNClusters) Some(computeSizes(model, data)) else None

    EvaluationResult(
      silhouette    = if (metrics.contains("silhouette")) Some(computeSilhouette(model, data, spec, knownTotalRows)) else None,
      nClusters     = if (wantNClusters) sizes.map(_.keys.count(_ >= 0)) else None,
      noiseFraction = if (wantNoise)     sizes.map(noiseFractionOf)      else None,
      clusterSizes  = if (wantSizes)     sizes                           else None
    )
  }

  /** Cluster id -> size, from a single collect. Cast prediction to Int so the
   *  read is independent of each model's label dtype. */
  private def computeSizes(model: Model, data: DataFrame): Map[Int, Long] =
    model.assignClusters(data)
      .groupBy(col(Columns.Prediction).cast(IntegerType).as(Columns.Prediction))
      .count()
      .collect()
      .map(r => r.getInt(0) -> r.getLong(1))
      .toMap

  /** Fraction of points labelled noise (cluster id -1). */
  private def noiseFractionOf(sizes: Map[Int, Long]): Double = {
    val totalLabeled = sizes.values.sum
    val noiseCount   = sizes.getOrElse(-1, 0L)
    if (totalLabeled == 0L) 0.0 else noiseCount.toDouble / totalLabeled
  }

  /** Silhouette on the full data set or a uniform sample, depending on spec.
   *  Sampling is essential for big data — the underlying evaluator is O(n^2). */
  private def computeSilhouette(model: Model, data: DataFrame, spec: EvaluationSpec, knownTotalRows: Option[Long]): Double = {
    val evaluator = new SilhouetteEvaluator(distance) // TODO: refactor this part of the code
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

/** Evaluation outputs. Every field is optional: only the metrics named in
 *  `EvaluationSpec.metrics` are computed, the rest stay None (and are omitted
 *  from the emitted RunResult). */
final case class EvaluationResult(
  silhouette:    Option[Double],
  nClusters:     Option[Int],
  noiseFraction: Option[Double],
  clusterSizes:  Option[Map[Int, Long]]
)

object EvaluationResult {
  /** No metrics computed — used for a failed run. */
  val empty: EvaluationResult = EvaluationResult(None, None, None, None)
}
