package clustering.benchmark

import clustering.benchmark.config.RunConfig
import clustering.benchmark.framework.SparkClusteringJob
import clustering.benchmark.metrics.RunResult
import org.log4s.getLogger


/** Entry point for one benchmark run.
 *
 *  Usage: `spark-submit ... benchmark.jar --config <path-to-run.json>`
 *
 *  Exactly one config file produces exactly one JSON result file under
 *  `<profile.outputDir>/<runId>.json`. Failure modes still write a result
 *  file (with `status: "failed"` and `errorMessage`) so SLURM array jobs
 *  never silently lose runs.
 */
object BenchmarkRunner {
  private val logger = getLogger

  def main(args: Array[String]): Unit = {
    val parsed = parseArgs(args)
    val config = RunConfig.fromFile(parsed.configPath)

    val profile = RunConfig.resolveProfile(config)
    val outputDir = RunConfig.resolveOutputDir(config, profile)

    val job = new SparkClusteringJob
    val result = job.run(config, profile)
    val path = RunResult.writeToDir(result, outputDir)

    logger.info(s"runId=${result.runId} status=${result.status} " +
      s"algo=${result.algorithm} fitMs=${result.fitDurationMs} " +
      s"totalMs=${result.totalDurationMs} nRows=${result.nRows} " +
      s"silhouette=${result.silhouette.getOrElse("n/a")} -> $path")

    if (result.status != "ok") sys.exit(2)
  }

  private case class Args(configPath: String)

  private def parseArgs(args: Array[String]): Args = {
    var configPath: Option[String] = None
    val it = args.iterator
    while (it.hasNext) {
      it.next() match {
        case "--config" => configPath = Some(it.next())
        case "--help" | "-h" =>
          println(usage()); sys.exit(0)
        case other =>
          logger.error(s"Unknown argument: $other\n${usage()}")
          sys.exit(64)
      }
    }
    Args(
      configPath = configPath.getOrElse {
        logger.error(s"--config is required\n${usage()}")
        sys.exit(64)
      }
    )
  }

  private def usage(): String =
    """Usage: spark-submit ... benchmark.jar --config <path>
      |
      |  --config  Path to a per-run JSON config (required)
      |""".stripMargin
}