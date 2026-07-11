package clustering.benchmark

import clustering.benchmark.config.{ClusterProfile, RunConfig}
import clustering.benchmark.datasource.DataSource
import clustering.benchmark.evaluation.{EvaluationResult, EvaluationRunner}
import clustering.benchmark.metrics.{BenchmarkListener, ProcessCpuPlugin, RunResult}
import clustering.benchmark.registry.{AlgorithmRegistry, DataSourceRegistry}
import clustering.core.{Clusterer, Columns}
import org.apache.spark.SparkConf
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.storage.StorageLevel
import org.log4s.getLogger

import java.io.File
import java.time.Instant
import scala.util.control.NonFatal
import scala.util.{Failure, Success, Try}

/** Runs one benchmark on Spark.
 *
 *  Pipeline: build SparkSession → install listener → load → fit → evaluate
 *  → assemble RunResult. The listener captures execution-level metrics
 *  (shuffle, GC, peak exec memory), wall-clock timings come from
 *  driver-side `System.nanoTime` around each phase.
 */
final class SparkClusteringJob {
  private val logger = getLogger

  val framework: String = "spark"

  def run(config: RunConfig, profile: ClusterProfile): RunResult = {
    val startedAt = Instant.now()
    val runStart  = System.nanoTime()

    logger.info(s"starting runId=${config.runId} algorithm=${config.algorithm.name} " +
      s"dataset=${config.dataset.`type`} profile=${profile.name}")

    val spark = buildSession(config, profile)
    val sc    = spark.sparkContext
    sc.setLogLevel("WARN")

    // GraphFrames' connected-components (used by DBSCAN) checkpoints its
    // iterations, so a checkpoint dir must be set before fit. Deleted below so
    // array-job matrices don't fill tmp with stale checkpoints.
    val checkpointDir = s"${System.getProperty("java.io.tmpdir")}/spark-checkpoints-${config.runId}"
    sc.setCheckpointDir(checkpointDir)

    val listener = new BenchmarkListener
    sc.addSparkListener(listener)
    ProcessCpuPlugin.reset()  // clear process-CPU accumulator before this run

    // Run all Spark-dependent work; Try captures a NonFatal failure as a value,
    // so no mutable success/failure flags. Timings are driver-side wall clock,
    // independent of the listener bus.
    val outcome = Try(execute(spark, config))
    val totalMs = elapsedMs(runStart, System.nanoTime())   // measured before stop(), like each phase

    // Assemble the RunResult only AFTER spark.stop(): SparkContext.stop()
    // synchronously drains the async listener bus, so reading listener/plugin
    // metrics before it would undercount tail events (tasks, stages).
    spark.stop()
    deleteRecursively(new File(checkpointDir))
    logger.info(s"runId=${config.runId} total=${totalMs}ms")

    val snap          = listener.snapshot()
    val process       = readProcessMetrics()
    val finishedAtIso = Instant.now().toString

    outcome match {
      case Success(e) =>
        RunResult.from(
          config, framework, profile.name,
          startedAtIso  = startedAt.toString,
          finishedAtIso = finishedAtIso,
          status        = "ok",
          errorMessage  = None,
          workload      = e.workload,
          timings       = e.timings.copy(totalMs = totalMs),
          snap          = snap,
          process       = process,
          eval          = e.eval
        )
      case Failure(t) =>
        logger.error(t)(s"run failed: ${t.getClass.getSimpleName}: ${t.getMessage}")
        RunResult.from(
          config, framework, profile.name,
          startedAtIso  = startedAt.toString,
          finishedAtIso = finishedAtIso,
          status        = "failed",
          errorMessage  = Some(s"${t.getClass.getSimpleName}: ${t.getMessage}"),
          workload      = RunResult.Workload.empty(config),
          timings       = RunResult.Timings(0L, 0L, 0L, totalMs),
          snap          = snap,
          process       = process,
          eval          = EvaluationResult.empty
        )
    }
  }

  /** Load → fit → evaluate, returning an immutable [[SparkClusteringJob.ExecutionResult]].
   *  Owns nothing beyond the phases; `run` wraps this in a Try and controls the
   *  session lifecycle + metric reads. `totalMs` is left 0 here — it spans the
   *  whole session and is filled in by `run`. */
  private def execute(spark: SparkSession, config: RunConfig): SparkClusteringJob.ExecutionResult = {
    val datasource: DataSource = DataSourceRegistry.create(config.dataset)
    val built                  = AlgorithmRegistry.create(config.algorithm)
    val clusterer: Clusterer   = built.clusterer
    val evalDistance           = built.distance   // same metric the algorithm was built with

    logger.info(s"loading dataset ${datasource.name}")
    val loadStart   = System.nanoTime()
    val data        = datasource.load(spark).persist(StorageLevel.MEMORY_AND_DISK)
    val nRows       = data.count()   // materialises the cache
    val loadEnd     = System.nanoTime()

    // After the timing window: cache is warm, so take(1)/getNumPartitions add
    // negligible cost and don't inflate loadMs.
    val nFeatures   = inferFeatureCount(data)
    val nPartitions = data.rdd.getNumPartitions
    logger.info(s"loaded nRows=$nRows nPartitions=$nPartitions nFeatures=${nFeatures.getOrElse("?")} " +
      s"in ${elapsedMs(loadStart, loadEnd)}ms")

    logger.info(s"fitting ${config.algorithm.name}")
    val fitStart = System.nanoTime()
    val model    = clusterer.fit(data)
    val fitEnd   = System.nanoTime()
    logger.info(s"fitted ${config.algorithm.name} in ${elapsedMs(fitStart, fitEnd)}ms")

    logger.info(s"evaluating (metrics=${config.evaluation.metrics.mkString(",")} " +
      s"sampleSize=${config.evaluation.sampleSize.getOrElse("full")})")
    val evalStart = System.nanoTime()
    val eval      = new EvaluationRunner(evalDistance)
      .run(model, data, config.evaluation, knownTotalRows = Some(nRows))
    val evalEnd   = System.nanoTime()
    logger.info(s"evaluated nClusters=${eval.nClusters.getOrElse("n/a")} " +
      s"silhouette=${eval.silhouette.map(s => f"$s%.4f").getOrElse("n/a")} " +
      s"noiseFraction=${eval.noiseFraction.map(v => f"$v%.4f").getOrElse("n/a")} in ${elapsedMs(evalStart, evalEnd)}ms")

    data.unpersist(blocking = true)   // sync: free blocks before reading memory metrics

    SparkClusteringJob.ExecutionResult(
      workload = RunResult.Workload(
        dataset = datasource.name, datasetMetadata = datasource.metadata,
        nRows = nRows, nPartitions = nPartitions, nFeatures = nFeatures),
      eval    = eval,
      timings = RunResult.Timings(
        loadMs  = elapsedMs(loadStart, loadEnd),
        fitMs   = elapsedMs(fitStart, fitEnd),
        evalMs  = elapsedMs(evalStart, evalEnd),
        totalMs = 0L)
    )
  }

  /** Spark configs that improve the fidelity of metrics arriving in our
   *  `BenchmarkListener`. Applied before per-run `sparkConf` so users can
   *  still override any of them.
   */
  private val MetricsConfDefaults: Map[String, String] = Map(
    "spark.executor.processTreeMetrics.enabled" -> "true",
    "spark.eventLog.logStageExecutorMetrics"    -> "true",
    "spark.executor.metrics.pollingInterval"    -> "1000",
    // Whole-JVM process CPU per executor, comparable with Flink's Status.JVM.CPU.Time.
    "spark.plugins"                             -> "clustering.benchmark.metrics.ProcessCpuPlugin"
  )

  private def buildSession(config: RunConfig, profile: ClusterProfile): SparkSession = {
    val conf = new SparkConf().setAppName(s"benchmark-${config.runId}")
    profile.sparkMaster.foreach(conf.setMaster)
    MetricsConfDefaults.foreach { case (k, v) => conf.set(k, v) }
    config.spark_config.foreach { case (k, v) => conf.set(k, v) }
    SparkSession.builder().config(conf).getOrCreate()
  }

  /** Infers the feature-vector dimensionality from a sample of one row — cheap,
   *  doesn't force the whole pipeline to materialize. */
  private def inferFeatureCount(df: DataFrame): Option[Int] = {
    val sample = df.take(1)
    if (sample.isEmpty) None else Some(sample.head.getAs[Vector](Columns.Features).size)
  }

  /** Milliseconds between two `System.nanoTime` readings. */
  private def elapsedMs(startNanos: Long, endNanos: Long): Long = (endNanos - startNanos) / 1000000L

  /** Best-effort recursive delete of the run's checkpoint dir. */
  private def deleteRecursively(f: File): Unit = {
    try {
      if (f.isDirectory) Option(f.listFiles()).foreach(_.foreach(deleteRecursively))
      f.delete()
    } catch { case NonFatal(e) => logger.warn(s"could not delete ${f.getPath}: ${e.getMessage}") }
  }

  /** Read the process-level metrics once (CPU + heap), for the RunResult. */
  private def readProcessMetrics(): RunResult.ProcessMetrics =
    RunResult.ProcessMetrics(
      cpuNanos           = ProcessCpuPlugin.totalCpuNanos(),
      maxWorkerHeapBytes = ProcessCpuPlugin.maxWorkerHeapBytes(),
      totalHeapBytes     = ProcessCpuPlugin.totalHeapBytes())
}

object SparkClusteringJob {

  /** Immutable output of the load→fit→evaluate phases, carried past `spark.stop()`
   *  so the final RunResult can be assembled once listener metrics are complete.
   *  `timings.totalMs` is filled by `run` (spans the whole session). */
  private final case class ExecutionResult(
    workload: RunResult.Workload,
    eval:     EvaluationResult,
    timings:  RunResult.Timings
  )
}
