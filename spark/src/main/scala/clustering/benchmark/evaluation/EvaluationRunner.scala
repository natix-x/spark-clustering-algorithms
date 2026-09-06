package clustering.benchmark.evaluation

import clustering.benchmark.config.EvaluationSpec
import clustering.core.{Columns, Model, Weights}
import clustering.distance.DistanceMetric
import clustering.evaluation.{CalinskiHarabaszIndex, ClusterMoments, DaviesBouldinIndex, SilhouetteEvaluator}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType

/** Post-fit metrics.
 *
 *  `clusterSizes`/`noiseFraction` count ROWS, not mass ([[clustering.core.Weights]]) — known gap,
 *  fix when `coreset` lands. The three quality indices (`silhouette`, `daviesBouldin`,
 *  `calinskiHarabasz`) aggregate MASS, and the silhouette's subsample is mass-proportional too.
 *
 *  Each of the three blocks below scans the full dataset independently (label stats = 1 pass,
 *  centroid moments = 2, silhouette = 1) because nothing caches `Model.assignClusters`'s output.
 */
final class EvaluationRunner(distance: DistanceMetric) {
  import EvaluationRunner.Metric

  def run(model: Model, data: DataFrame, spec: EvaluationSpec): EvaluationResult = {
    val requested = spec.metrics.toSet
    validateMetrics(requested)

    // clusterSizes, noiseFraction, nClusters, silhouette share a single assignment scan.
    val needsStats = requested.intersect(Set(Metric.ClusterSizes, Metric.NoiseFraction, Metric.NClusters, Metric.Silhouette)).nonEmpty
    val stats = if (needsStats) Some(computeStats(model, data)) else None

    // Both centroid indices are derived from the SAME per-cluster moments.
    val needsMoments = requested.intersect(Set(Metric.DaviesBouldin, Metric.CalinskiHarabasz)).nonEmpty
    val moments = if (needsMoments) Some(ClusterMoments.compute(model, data, distance)) else None

    val silhouette = if (requested(Metric.Silhouette)) Some(computeSilhouette(model, data, spec, stats)) else None

    EvaluationResult(
      silhouette = silhouette.map(_.score),
      silhouetteScoredPoints = silhouette.map(_.scoredPoints),
      silhouetteSampleClusters = silhouette.map(_.sampleClusters),
      silhouetteUnscoredPoints = silhouette.map(_.unscoredPoints),
      // Emitted whenever the scan ran at all, because `silhouetteSampleClusters` is unreadable without it.
      nClusters = stats.map(_.nClusters),
      noiseFraction = if (requested(Metric.NoiseFraction)) stats.map(_.noiseFraction) else None,
      clusterSizes = if (requested(Metric.ClusterSizes)) stats.map(_.rows) else None,
      daviesBouldin = if (requested(Metric.DaviesBouldin)) moments.map(DaviesBouldinIndex.of(_, distance)) else None,
      calinskiHarabasz = if (requested(Metric.CalinskiHarabasz)) moments.map(CalinskiHarabaszIndex.of(_, distance)) else None
    )
  }

  private def validateMetrics(requested: Set[String]): Unit = {
    val unknown = requested -- Metric.All
    require(
      unknown.isEmpty,
      s"Unknown evaluation metric(s): ${unknown.toSeq.sorted.mkString(", ")}; " +
        s"Known: ${Metric.All.toSeq.sorted.mkString(", ")}"
    )
  }

  /** Cluster id -> (row count, mass), from a single scan. Prediction cast to Int for a
   *  dtype-independent read; mass uses `safeColumn` (not `column`) so a corrupt weight can't make
   *  a cluster's mass NULL/NaN/negative. */
  private def computeStats(model: Model, data: DataFrame): LabelStats = {
    val labeled = model.assignClusters(data)
    val aggregated = labeled
      .groupBy(col(Columns.Prediction).cast(IntegerType).as(Columns.Prediction))
      .agg(count(lit(1)).as("rows"), sum(Weights.safeColumn(labeled)).as("mass"))
      .collect()

    val rowMap = aggregated.map(r => r.getInt(0) -> r.getLong(1)).toMap
    val massMap = aggregated.map(r => r.getInt(0) -> (if (r.isNullAt(2)) 0.0 else r.getDouble(2))).toMap

    LabelStats(rowMap, massMap)
  }

  /** Silhouette on the full dataset or a subsample, plus counters saying what it was actually computed on. */
  private def computeSilhouette(
    model: Model,
    data: DataFrame,
    spec: EvaluationSpec,
    stats: Option[LabelStats]
  ): SilhouetteEvaluator.Outcome = {
    val population = stats.map(s => SilhouetteEvaluator.Population(s.labeledMass))
    val labeledRows = stats.map(_.labeledRowCount).getOrElse(Long.MaxValue)

    require(
      spec.sampleSize.exists(_ > 0) || labeledRows <= EvaluationRunner.FullSilhouetteMaxRows,
      s"Silhouette requested with no sampleSize on $labeledRows labelled rows, above the " +
        s"${EvaluationRunner.FullSilhouetteMaxRows} the driver can collect: an unsampled " +
        "silhouette collects every labelled row and is O(n^2 * d). Set evaluation.sampleSize " +
        "(10000 is the default) or run this config on a smaller dataset."
    )

    new SilhouetteEvaluator(distance).measure(model, data, spec.sampleSize, spec.seed, population)
  }
}

/** Per-label row counts and weight masses, from one assignment scan. Equal maps on unweighted
 *  input; `mass` is what the weight-aware metrics need. */
private[evaluation] final case class LabelStats(rows: Map[Int, Long], mass: Map[Int, Double]) {

  /** Number of clusters (excluding noise). */
  def nClusters: Int = rows.keys.count(_ >= 0)

  /** Fraction of points labelled noise (cluster id -1). */
  def noiseFraction: Double = {
    val totalLabeled = rows.values.sum
    if (totalLabeled == 0L) 0.0 else rows.getOrElse(-1, 0L).toDouble / totalLabeled
  }

  /** Total row count excluding noise. */
  def labeledRowCount: Long = rows.filter(_._1 >= 0).values.sum

  /** Mass mapped by cluster ID, excluding noise. */
  def labeledMass: Map[Int, Double] = mass.filter(_._1 >= 0)
}

object EvaluationRunner {
  object Metric {
    val Silhouette = "silhouette"
    val NClusters = "nClusters"
    val ClusterSizes = "clusterSizes"
    val NoiseFraction = "noiseFraction"
    val DaviesBouldin = "daviesBouldin"
    val CalinskiHarabasz = "calinskiHarabasz"

    val All: Set[String] = Set(Silhouette, NClusters, ClusterSizes, NoiseFraction, DaviesBouldin, CalinskiHarabasz)
  }

  /** Most labelled ROWS an unsampled (`sampleSize: null`) silhouette may collect to the driver.
   *  Rationale in [[EvaluationRunner.computeSilhouette]]; rows rather than mass because rows are
   *  what the driver holds. */
  val FullSilhouetteMaxRows: Long = 50000L
}

/** Evaluation outputs. Every field is optional: only the metrics named in
 *  `EvaluationSpec.metrics` are computed, the rest stay None (and are omitted
 *  from the emitted RunResult). */
final case class EvaluationResult(
  silhouette: Option[Double],
  silhouetteScoredPoints: Option[Int],
  silhouetteSampleClusters: Option[Int],
  silhouetteUnscoredPoints: Option[Int],
  nClusters: Option[Int],
  noiseFraction: Option[Double],
  clusterSizes: Option[Map[Int, Long]],
  daviesBouldin: Option[Double],
  calinskiHarabasz: Option[Double]
)

object EvaluationResult {
  /** No metrics computed — used for a failed run. */
  val empty: EvaluationResult =
    EvaluationResult(None, None, None, None, None, None, None, None, None)
}