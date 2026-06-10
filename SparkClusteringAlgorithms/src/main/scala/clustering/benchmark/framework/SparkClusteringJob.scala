package clustering.benchmark.framework

import clustering.benchmark.config.{ClusterProfile, RunConfig}
import clustering.benchmark.datasource.DataSource
import clustering.benchmark.evaluation.EvaluationRunner
import clustering.benchmark.metrics.BenchmarkListener.ListenerSnapshot
import clustering.benchmark.metrics.{BenchmarkListener, RunResult}
import clustering.benchmark.registry.{AlgorithmRegistry, DistanceRegistry}
import clustering.core.Clusterer
import clustering.distance.{DistanceMetric, EuclideanDistance}
import org.apache.spark.SparkConf
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.json4s._
import org.log4s.getLogger

import java.time.Instant

/** Spark-backed implementation of `ClusteringJob`.
 *
 *  Pipeline: build SparkSession → install listener → load → fit → evaluate
 *  → assemble RunResult. The listener captures execution-level metrics
 *  (shuffle, GC, peak exec memory), wall-clock timings come from
 *  driver-side `System.nanoTime` around each phase.
 */
final class SparkClusteringJob extends ClusteringJob {
  private val logger = getLogger

  override val framework: String = "spark"

  override def run(config: RunConfig, profile: ClusterProfile): RunResult = {
    val startedAt = Instant.now()
    val t0Total   = System.nanoTime()

    logger.info(s"starting runId=${config.runId} algorithm=${config.algorithm.name} " +
      s"dataset=${config.dataset.`type`} profile=${profile.name}")

    val spark = buildSession(config, profile)
    val sc    = spark.sparkContext
    sc.setLogLevel("WARN")

    // GraphFrames' connected-components (used by DBSCAN) checkpoints its
    // iterations, so a checkpoint dir must be set before fit.
    sc.setCheckpointDir(s"${System.getProperty("java.io.tmpdir")}/spark-checkpoints-${config.runId}")

    val listener = new BenchmarkListener
    sc.addSparkListener(listener)

    try {
      val datasource: DataSource = DataSource.create(config.dataset)
      val clusterer: Clusterer   = AlgorithmRegistry.create(config.algorithm)
      val evalDistance           = evaluationDistance(config)

      // --- load ---------------------------------------------------------
      logger.info(s"loading dataset ${datasource.name}")
      val t0Load      = System.nanoTime()
      val data        = datasource.load(spark)

      // inferDim before count() so dim inference doesn't add to load time
      val nFeatures   = inferDim(data)
      val nRows       = data.count()
      val nPartitions = data.rdd.getNumPartitions
      val t1Load      = System.nanoTime()
      logger.info(s"loaded nRows=$nRows nPartitions=$nPartitions nFeatures=${nFeatures.getOrElse("?")} " +
        s"in ${ms(t0Load, t1Load)}ms")

      // --- fit ----------------------------------------------------------
      logger.info(s"fitting ${config.algorithm.name}")
      val t0Fit  = System.nanoTime()
      val model  = clusterer.fit(data)
      val t1Fit  = System.nanoTime()
      logger.info(s"fitted ${config.algorithm.name} in ${ms(t0Fit, t1Fit)}ms")

      // --- evaluate -----------------------------------------------------
      logger.info(s"evaluating (metrics=${config.evaluation.metrics.mkString(",")} " +
        s"sampleSize=${config.evaluation.sampleSize.getOrElse("full")})")
      val t0Eval = System.nanoTime()
      val eval   = new EvaluationRunner(evalDistance)
        .run(model, data, config.evaluation, knownTotalRows = Some(nRows))
      val t1Eval = System.nanoTime()
      logger.info(s"evaluated nClusters=${eval.nClusters} " +
        s"silhouette=${eval.silhouette.map(s => f"$s%.4f").getOrElse("n/a")} " +
        s"noiseFraction=${f"${eval.noiseFraction}%.4f"} in ${ms(t0Eval, t1Eval)}ms")

      val finishedAt = Instant.now()
      val snap       = listener.snapshot()

      RunResult(
        runId                   = config.runId,
        framework               = framework,
        profile                 = profile.name,
        startedAtIso            = startedAt.toString,
        finishedAtIso           = finishedAt.toString,
        status                  = "ok",
        errorMessage            = None,
        algorithm               = config.algorithm.name,
        algorithmParams         = flatten(config.algorithm.params),
        dataset                 = datasource.name,
        datasetMetadata         = datasource.metadata,
        sparkConf               = config.sparkConf,
        experimentMetadata      = config.experimentMetadata,
        nRows                   = nRows,
        nPartitions             = nPartitions,
        nFeatures               = nFeatures,
        loadDurationMs          = ms(t0Load, t1Load),
        fitDurationMs           = ms(t0Fit, t1Fit),
        evalDurationMs          = ms(t0Eval, t1Eval),
        totalDurationMs         = ms(t0Total, System.nanoTime()),
        shuffleReadBytes        = snap.shuffleReadBytes,
        shuffleWriteBytes       = snap.shuffleWriteBytes,
        inputBytes              = snap.inputBytes,
        outputBytes             = snap.outputBytes,
        diskBytesSpilled        = snap.diskBytesSpilled,
        memoryBytesSpilled      = snap.memoryBytesSpilled,
        jvmGcTimeMs             = snap.jvmGcTimeMs,
        executorCpuTimeNs       = snap.executorCpuTimeNs,
        executorRunTimeMs       = snap.executorRunTimeMs,
        shuffleFetchWaitTimeMs  = snap.shuffleFetchWaitTimeMs,
        shuffleWriteTimeNs      = snap.shuffleWriteTimeNs,
        taskCount               = snap.taskCount,
        failedTaskCount         = snap.failedTaskCount,
        stageCount               = snap.stageCount,
        totalStageMs            = snap.totalStageMs,
        peakExecutorMemoryBytes = snap.peakExecutorMemoryBytes,
        nClusters               = eval.nClusters,
        noiseFraction           = eval.noiseFraction,
        silhouette              = eval.silhouette,
        clusterSizes            = eval.clusterSizes.map { case (k, v) => k.toString -> v }
      )
    } catch {
      case t: Throwable =>
        logger.error(t)(s"run failed: ${t.getClass.getSimpleName}: ${t.getMessage}")
        val snap = listener.snapshot()
        failedResult(config, profile, startedAt, t0Total, snap, t)
    } finally {
      spark.stop()
      logger.info(s"runId=${config.runId} total=${ms(t0Total, System.nanoTime())}ms")
    }
  }

  /** Spark configs that improve the fidelity of metrics arriving in our
   *  `BenchmarkListener`. Applied before per-run `sparkConf` so users can
   *  still override any of them.
   */
  private val MetricsConfDefaults: Map[String, String] = Map(
    "spark.executor.processTreeMetrics.enabled" -> "true",
    "spark.eventLog.logStageExecutorMetrics"    -> "true",
    "spark.executor.metrics.pollingInterval"    -> "1000"
  )

  private def buildSession(config: RunConfig, profile: ClusterProfile): SparkSession = {
    val conf = new SparkConf().setAppName(s"benchmark-${config.runId}")
    profile.sparkMaster.foreach(conf.setMaster)
    MetricsConfDefaults.foreach { case (k, v) => conf.set(k, v) }
    config.sparkConf.foreach    { case (k, v) => conf.set(k, v) }
    SparkSession.builder().config(conf).getOrCreate()
  }

  private def evaluationDistance(config: RunConfig): DistanceMetric =
    (config.algorithm.params \ "distance") match {
      case JString(name) => DistanceRegistry.get(name)
      case _             => EuclideanDistance
    }

  /** Infers dimension from a sample of one row — cheap, doesn't force the
   *  whole pipeline to materialize. */
  private def inferDim(df: DataFrame): Option[Int] = {
    val sample = df.take(1)
    if (sample.isEmpty) None else Some(sample.head.getAs[Vector]("features").size)
  }

  private def ms(start: Long, end: Long): Long = (end - start) / 1000000L

  /** json4s JObject -> Map[String, String] for the flat result schema. */
  private def flatten(params: JObject): Map[String, String] = {
    params.obj.map { case (k, v) =>
      k -> (v match {
        case JString(s)   => s
        case JBool(b)     => b.toString
        case JInt(n)      => n.toString
        case JLong(n)     => n.toString
        case JDouble(d)   => d.toString
        case JDecimal(d)  => d.toString
        case JNull        => "null"
        case other        => org.json4s.jackson.JsonMethods.compact(org.json4s.jackson.JsonMethods.render(other))
      })
    }.toMap
  }

  private def failedResult(
    config:    RunConfig,
    profile:   ClusterProfile,
    startedAt: Instant,
    t0Total:   Long,
    snap:      ListenerSnapshot,
    error:     Throwable
  ): RunResult = RunResult(
    runId                   = config.runId,
    framework               = framework,
    profile                 = profile.name,
    startedAtIso            = startedAt.toString,
    finishedAtIso           = Instant.now().toString,
    status                  = "failed",
    errorMessage            = Some(s"${error.getClass.getSimpleName}: ${error.getMessage}"),
    algorithm               = config.algorithm.name,
    algorithmParams         = flatten(config.algorithm.params),
    dataset                 = config.dataset.`type`,
    datasetMetadata         = Map.empty,
    sparkConf               = config.sparkConf,
    experimentMetadata      = config.experimentMetadata,
    nRows                   = -1L,
    nPartitions             = -1,
    nFeatures               = None,
    loadDurationMs          = 0L,
    fitDurationMs           = 0L,
    evalDurationMs          = 0L,
    totalDurationMs         = ms(t0Total, System.nanoTime()),
    shuffleReadBytes        = snap.shuffleReadBytes,
    shuffleWriteBytes       = snap.shuffleWriteBytes,
    inputBytes              = snap.inputBytes,
    outputBytes             = snap.outputBytes,
    diskBytesSpilled        = snap.diskBytesSpilled,
    memoryBytesSpilled      = snap.memoryBytesSpilled,
    jvmGcTimeMs             = snap.jvmGcTimeMs,
    executorCpuTimeNs       = snap.executorCpuTimeNs,
    executorRunTimeMs       = snap.executorRunTimeMs,
    shuffleFetchWaitTimeMs  = snap.shuffleFetchWaitTimeMs,
    shuffleWriteTimeNs      = snap.shuffleWriteTimeNs,
    taskCount               = snap.taskCount,
    failedTaskCount         = snap.failedTaskCount,
    stageCount              = snap.stageCount,
    totalStageMs            = snap.totalStageMs,
    peakExecutorMemoryBytes = snap.peakExecutorMemoryBytes,
    nClusters               = 0,
    noiseFraction           = 0.0,
    silhouette              = None,
    clusterSizes            = Map.empty
  )
}