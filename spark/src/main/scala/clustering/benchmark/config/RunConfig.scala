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
 *  @param profile     Cluster profile name ("local" | "ares"); defaults to "local" if absent.
 *  @param dataset     DataSource spec (type + params).
 *  @param algorithm   Algorithm spec (name + params).
 *  @param evaluation  Which metrics to compute, optionally subsampled.
 *  @param spark_config Extra SparkConf entries set programmatically before session start.
 *                     Snake_case to match the input contract key (see run_config.schema.json).
 *  @param outputDir   Where to write the result; a local-dev default is used if absent.
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
  spark_config: Map[String, String] = Map.empty,
  outputDir: Option[String] = None,
  experimentMetadata: Map[String, String] = Map.empty
)

final case class DataSourceSpec(`type`: String, params: JObject)

final case class AlgorithmSpec(name: String, params: JObject)

final case class EvaluationSpec(
  metrics: Seq[String] = Seq("silhouette", "nClusters", "clusterSizes", "noiseFraction"),
  sampleSize: Option[Int] = None,
  seed: Long = 42L
)

object RunConfig {

  private implicit val formats: Formats = DefaultFormats

  def fromFile(path: String): RunConfig =
    parse(Files.readString(Paths.get(path))).extract[RunConfig]

  def resolveProfile(cfg: RunConfig): ClusterProfile =
    cfg.profile.map(ClusterProfile.fromName).getOrElse(LocalProfile)

  private val DefaultOutputDir: String =
    sys.props.getOrElse("user.dir", ".") + "/benchmark-results"

  def resolveOutputDir(cfg: RunConfig): String =
    cfg.outputDir.getOrElse(DefaultOutputDir)
}
