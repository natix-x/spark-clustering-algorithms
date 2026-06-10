package clustering.benchmark.framework

import clustering.benchmark.config.{ClusterProfile, RunConfig}
import clustering.benchmark.metrics.RunResult

/** Framework-agnostic seam: one impl per Big Data platform (Spark, Flink, ...).
 *
 *  The BenchmarkRunner only knows this trait, so adding Flink later is a new
 *  impl + a one-line dispatch — no changes to config, runner, or result schema.
 */
trait ClusteringJob extends Serializable {
  def framework: String
  def run(config: RunConfig, profile: ClusterProfile): RunResult
}