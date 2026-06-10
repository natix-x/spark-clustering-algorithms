package clustering.benchmark.config

import org.json4s._
import org.json4s.jackson.JsonMethods._

import java.nio.file.{Files, Paths}

/** Root config for a single benchmark run.
 *
 *  Mirrors a hand-edited or matrix-generated per-run JSON file. Algorithm and
 *  dataset params are kept as raw `JObject`s so each factory parses what it
 *  needs without a god-class case-class hierarchy.
 *
 *  @param runId       Globally unique id; becomes the result filename stem.
 *  @param profile     Cluster profile name ("local" | "ares"); env overrides if absent.
 *  @param dataset     DataSource spec (type + params).
 *  @param algorithm   Algorithm spec (name + params).
 *  @param evaluation  Which metrics to compute, optionally subsampled.
 *  @param sparkConf   Extra SparkConf entries set programmatically before session start.
 *  @param outputDir   Overrides profile.outputDir when present.
 *  @param experimentMetadata Freeform key→value pairs describing the experiment
 *                     setup (nodes, cores, memory, walltime, ...); written verbatim
 *                     into the result JSON for analysis.
 */
final case class RunConfig(
  runId: String,
  profile: Option[String],
  dataset: DataSourceSpec,
  algorithm: AlgorithmSpec,
  evaluation: EvaluationSpec,
  sparkConf: Map[String, String] = Map.empty,
  outputDir: Option[String] = None,
  experimentMetadata: Map[String, String] = Map.empty
)

final case class DataSourceSpec(`type`: String, params: JObject)

final case class AlgorithmSpec(name: String, params: JObject)

/** What to evaluate after the model is fit.
 *
 *  @param metrics    Subset of {"silhouette", "clusterSizes", "noiseFraction"}.
 *  @param sampleSize Optional cap for O(n^2) metrics like silhouette;
 *                    None = run on full dataset (only feasible for small data).
 */
final case class EvaluationSpec(
  metrics: Seq[String] = Seq("silhouette", "clusterSizes", "noiseFraction"),
  sampleSize: Option[Int] = None,
  seed: Long = 42L
)

object RunConfig {

  private implicit val formats: Formats = DefaultFormats

  def fromJsonString(json: String): RunConfig = parse(json).extract[RunConfig]

  def fromFile(path: String): RunConfig = {
    val bytes = Files.readAllBytes(Paths.get(path))
    fromJsonString(new String(bytes, "UTF-8"))
  }

  /** Resolve the effective profile: explicit field beats env auto-detection. */
  def resolveProfile(cfg: RunConfig): ClusterProfile =
    cfg.profile.map(ClusterProfile.fromName).getOrElse(ClusterProfile.fromEnv())

  /** Resolve effective output directory: per-run override beats profile default. */
  def resolveOutputDir(cfg: RunConfig, profile: ClusterProfile): String =
    cfg.outputDir.getOrElse(profile.outputDir)
}