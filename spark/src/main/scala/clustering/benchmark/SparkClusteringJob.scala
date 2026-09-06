package clustering.benchmark

import clustering.benchmark.config.{ClusterProfile, RunConfig}
import clustering.benchmark.evaluation.{EvaluationResult, EvaluationRunner}
import clustering.benchmark.metrics.{ProcessCpuPlugin, RunResult}
import clustering.benchmark.registry.{AlgorithmRegistry, DataSourceRegistry}
import clustering.core.Columns
import org.apache.spark.SparkConf
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.storage.StorageLevel
import org.log4s.getLogger

import java.time.Instant
import scala.util.{Failure, Success, Try}

/**
 * Executes a single clustering benchmark on Apache Spark.
 * Captures wall-clock timings and JVM resource metrics (CPU, GC, Heap) across phases: load, fit, eval.
 */
final class SparkClusteringJob {
  private val logger = getLogger

  val framework: String = "spark"

  def run(config: RunConfig, profile: ClusterProfile): RunResult = {
    logger.info(s"Starting runId=${config.runId} algorithm=${config.algorithm.name} " +
      s"dataset=${config.dataset.`type`} profile=${profile.name}")

    val spark = buildSession(config, profile)
    val sc = spark.sparkContext
    sc.setLogLevel("WARN")

    awaitExecutors(sc)

    // Settle interval ensures all executors publish their initial metrics before we baseline.
    val metricSettleTimeMs = ProcessCpuPlugin.settleMs(ProcessCpuPlugin.samplingIntervalMs(sc.getConf))
    ProcessCpuPlugin.reset()
    waitForMetricPublishers(metricSettleTimeMs, "baseline")
    ProcessCpuPlugin.captureBaseline()

    val timeline = new SparkClusteringJob.PhaseTimeline(metricSettleTimeMs, ProcessCpuPlugin.baseline)
    val startTimestampUtc = Instant.now()
    val startNanos = System.nanoTime()

    val executionResultTry = Try(executePhasePipeline(spark, config, timeline))
    val totalDurationMs = toMillis(startNanos, System.nanoTime())

    // Give samplers one last period to push tail metrics before shutting down executors.
    waitForMetricPublishers(metricSettleTimeMs, "shutdown")

    // Stopping context runs ExecutorPlugin's shutdown, forcing the final metric flush.
    spark.stop()
    logger.info(s"runId=${config.runId} totalDuration=${totalDurationMs}ms")

    val processMetrics = readProcessMetrics()
    val finishedTimestampUtc = Instant.now().toString

    executionResultTry match {
      case Success(result) =>
        RunResult.from(
          config,
          framework,
          profile.name,
          startedAtIso = startTimestampUtc.toString,
          finishedAtIso = finishedTimestampUtc,
          status = "ok",
          errorMessage = None,
          workload = result.workload,
          timings = result.timings.copy(totalMs = totalDurationMs),
          phases = RunResult.PhaseMetrics.from(timeline.deltas, result.phaseDurationsMs, timeline.peaks),
          process = processMetrics,
          eval = result.evaluationResult
        )

      case Failure(exception) =>
        logger.error(exception)(s"Run failed: ${exception.getClass.getSimpleName}: ${exception.getMessage}")
        RunResult.from(
          config,
          framework,
          profile.name,
          startedAtIso = startTimestampUtc.toString,
          finishedAtIso = finishedTimestampUtc,
          status = "failed",
          errorMessage = Some(s"${exception.getClass.getSimpleName}: ${exception.getMessage}"),
          workload = RunResult.Workload.empty(config),
          timings = RunResult.Timings(0L, 0L, 0L, totalDurationMs),
          phases = RunResult.PhaseMetrics.from(timeline.deltas, Map.empty, timeline.peaks),
          process = processMetrics,
          eval = EvaluationResult.empty
        )
    }
  }

  private def executePhasePipeline(
    spark: SparkSession,
    config: RunConfig,
    timeline: SparkClusteringJob.PhaseTimeline
  ): SparkClusteringJob.ExecutionResult = {
    val dataSource = DataSourceRegistry.create(config.dataset)
    val algorithmInstance = AlgorithmRegistry.create(config.algorithm)
    val clusterer = algorithmInstance.clusterer
    val evaluationMetric = algorithmInstance.distance

    // PHASE: LOAD
    logger.info(s"Loading dataset ${dataSource.name}")
    timeline.beginPhase()
    val loadStartNanos = System.nanoTime()

    val data = dataSource.load(spark).persist(StorageLevel.MEMORY_AND_DISK)
    val rowCount = data.count() // materializes the cache

    val loadEndNanos = System.nanoTime()
    val loadDurationMs = toMillis(loadStartNanos, loadEndNanos)
    val featureCount = inferFeatureCount(data)
    val partitionCount = data.rdd.getNumPartitions

    logger.info(s"Loaded rowCount=$rowCount partitionCount=$partitionCount " +
      s"featureCount=${featureCount.getOrElse("?")} in ${loadDurationMs}ms")
    timeline.closePhase("load")

    // PHASE: FIT
    logger.info(s"Fitting ${config.algorithm.name}")
    timeline.beginPhase()
    val fitStartNanos = System.nanoTime()

    val model = clusterer.fit(data)

    val fitEndNanos = System.nanoTime()
    val fitDurationMs = toMillis(fitStartNanos, fitEndNanos)

    logger.info(s"Fitted ${config.algorithm.name} in ${fitDurationMs}ms")
    timeline.closePhase("fit")

    // PHASE: EVALUATE
    logger.info(s"Evaluating (metrics=${config.evaluation.metrics.mkString(",")} " +
      s"sampleSize=${config.evaluation.sampleSize.getOrElse("full")})")
    timeline.beginPhase()
    val evalStartNanos = System.nanoTime()

    val evaluationResult = new EvaluationRunner(evaluationMetric).run(model, data, config.evaluation)

    val evalEndNanos = System.nanoTime()
    val evalDurationMs = toMillis(evalStartNanos, evalEndNanos)

    logger.info(s"Evaluated nClusters=${evaluationResult.nClusters.getOrElse("n/a")} " +
      s"silhouette=${evaluationResult.silhouette.map(s => f"$s%.4f").getOrElse("n/a")} " +
      s"in ${evalDurationMs}ms")
    timeline.closePhase("eval")

    // Teardown - unpersist data before reading memory metrics
    data.unpersist(blocking = true)

    SparkClusteringJob.ExecutionResult(
      workload = RunResult.Workload(
        dataset = dataSource.name,
        datasetMetadata = dataSource.metadata,
        nRows = rowCount,
        nPartitions = partitionCount,
        nFeatures = featureCount
      ),
      evaluationResult = evaluationResult,
      timings = RunResult.Timings(
        loadMs = loadDurationMs,
        fitMs = fitDurationMs,
        evalMs = evalDurationMs,
        totalMs = 0L // Populated in the outer run method
      ),
      phaseDurationsMs = Map(
        "load" -> loadDurationMs,
        "fit" -> fitDurationMs,
        "eval" -> evalDurationMs
      )
    )
  }

  /**
   * Blocks until expected executors are registered so timings reflect full allocation.
   */
  private def awaitExecutors(sc: org.apache.spark.SparkContext): Unit = {
    val awaitTimeoutMs = 120000L // 2 minutes

    val expectedExecutorCount = for {
      totalCores <- sc.getConf.getOption("spark.cores.max").map(_.toInt)
      coresPerExecutor <- sc.getConf.getOption("spark.executor.cores").map(_.toInt)
      if coresPerExecutor > 0
    } yield totalCores / coresPerExecutor

    expectedExecutorCount.filter(_ > 1).foreach { targetCount =>
      val deadlineNanos = System.nanoTime() + awaitTimeoutMs * 1000000L

      def registeredExecutors: Int = math.max(0, sc.getExecutorMemoryStatus.size - 1) // -1 for driver

      while (registeredExecutors < targetCount && System.nanoTime() < deadlineNanos) {
        Thread.sleep(500L)
      }

      if (registeredExecutors < targetCount) {
        logger.warn(s"Only $registeredExecutors/$targetCount executors registered within " +
          s"${awaitTimeoutMs}ms; proceeding at reduced parallelism.")
      } else {
        logger.info(s"Cluster ready: $registeredExecutors/$targetCount executors registered.")
      }
    }
  }

  /**
   * Required to allow background JVM threads on executors to push their sampled metrics to the driver.
   */
  private def waitForMetricPublishers(settleMs: Long, context: String): Unit = {
    logger.info(s"Waiting ${settleMs}ms for metric samplers to report ($context)")
    Thread.sleep(settleMs)
  }

  private val metricsConfDefaults: Map[String, String] = Map(
    "spark.plugins" -> "clustering.benchmark.metrics.ProcessCpuPlugin"
  )

  private def buildSession(config: RunConfig, profile: ClusterProfile): SparkSession = {
    val conf = new SparkConf().setAppName(s"benchmark-${config.runId}")
    profile.sparkMaster.foreach(conf.setMaster)
    metricsConfDefaults.foreach { case (k, v) => conf.set(k, v) }
    config.spark_config.foreach { case (k, v) => conf.set(k, v) }

    SparkSession.builder().config(conf).getOrCreate()
  }

  private def inferFeatureCount(df: DataFrame): Option[Int] = {
    val sample = df.take(1)
    if (sample.isEmpty) None else Some(sample.head.getAs[Vector](Columns.Features).size)
  }

  private def toMillis(startNanos: Long, endNanos: Long): Long =
    (endNanos - startNanos) / 1000000L

  private def readProcessMetrics(): RunResult.ProcessMetrics =
    RunResult.ProcessMetrics(
      cpuNanos = ProcessCpuPlugin.totalCpuNanos(),
      maxWorkerHeapBytes = ProcessCpuPlugin.maxWorkerHeapBytes(),
      totalHeapBytes = ProcessCpuPlugin.totalHeapBytes(),
      heapByteSeconds = ProcessCpuPlugin.totalHeapByteSeconds(),
      gcTimeMs = ProcessCpuPlugin.totalGcTimeMs(),
      maxWorkerDirectBytes = ProcessCpuPlugin.maxWorkerDirectBytes(),
      totalDirectBytes = ProcessCpuPlugin.totalDirectBytes(),
      driverCpuNanos = ProcessCpuPlugin.getDriverCpuNanos,
      driverPeakHeapBytes = ProcessCpuPlugin.getDriverPeakHeapBytes,
      driverHeapByteSeconds = ProcessCpuPlugin.getDriverHeapByteSeconds,
      driverGcTimeMs = ProcessCpuPlugin.getDriverGcTimeMs,
      driverPeakDirectBytes = ProcessCpuPlugin.getDriverPeakDirectBytes
    )
}

object SparkClusteringJob {

  private final case class ExecutionResult(
    workload: RunResult.Workload,
    evaluationResult: EvaluationResult,
    timings: RunResult.Timings,
    phaseDurationsMs: Map[String, Long]
  )

  /**
   * Tracks metrics boundaries across benchmark phases.
   * Not thread-safe, intended for sequential execution on the driver.
   */
  private final class PhaseTimeline(settleTimeMs: Long, initialSnapshot: ProcessCpuPlugin.CounterSnapshot) {
    private val logger = getLogger
    private val snapshots = scala.collection.mutable.ArrayBuffer[(String, ProcessCpuPlugin.CounterSnapshot)](
      "" -> initialSnapshot
    )

    def beginPhase(): Unit = ProcessCpuPlugin.beginPhase()

    def closePhase(phaseName: String): Unit = {
      logger.info(s"Waiting ${settleTimeMs}ms at the $phaseName boundary to collect metrics")
      Thread.sleep(settleTimeMs)

      snapshots += (phaseName -> ProcessCpuPlugin.snapshot())
      ProcessCpuPlugin.endPhase(phaseName)
    }

    def peaks: Seq[(String, ProcessCpuPlugin.PhasePeaks)] = ProcessCpuPlugin.windowPeaks

    def deltas: Seq[(String, ProcessCpuPlugin.PhaseDelta)] =
      snapshots.indices.drop(1).map { i =>
        snapshots(i)._1 -> ProcessCpuPlugin.between(snapshots(i - 1)._2, snapshots(i)._2)
      }
  }
}
