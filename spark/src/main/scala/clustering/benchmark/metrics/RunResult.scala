package clustering.benchmark.metrics

import clustering.benchmark.config.RunConfig
import clustering.benchmark.evaluation.EvaluationResult
import org.json4s._
import org.json4s.jackson.JsonMethods._

import java.nio.charset.StandardCharsets
import java.nio.file.{AtomicMoveNotSupportedException, Files, Path, Paths, StandardCopyOption, StandardOpenOption}

/**
 * One row of benchmark output. Flat by design so analysis in pandas is
 * a single `pd.read_json` over the results directory.
 * Nested structures are kept as maps and flattened later by the merger script.
 */
final case class RunResult(
  runId: String,
  framework: String,
  profile: String,
  startedAtIso: String,
  finishedAtIso: String,
  status: String,
  errorMessage: Option[String],

  algorithm: String,
  algorithmParams: Map[String, String],
  dataset: String,
  datasetMetadata: Map[String, String],
  sparkConf: Map[String, String],
  experimentMetadata: Map[String, String],

  nRows: Long,
  nPartitions: Int,
  nFeatures: Option[Int],

  // Wall-clock timings (driver, ms)
  loadDurationMs: Long,
  fitDurationMs: Long,
  evalDurationMs: Long,
  totalDurationMs: Long,

  /**
   * Per-phase breakdown ("load", "fit", "eval").
   * Excludes post-eval teardown (like cache unpersist or executor shutdown) to ensure
   * the eval metrics only reflect the algorithm's actual execution and memory cost.
   */
  phaseDurationsMs: Map[String, Long],
  phaseCpuTimeNs: Map[String, Long],
  phaseGcTimeMs: Map[String, Long],
  phaseMemoryGbHours: Map[String, Double],
  phaseDriverCpuTimeNs: Map[String, Long],
  phaseDriverGcTimeMs: Map[String, Long],
  phaseDriverMemoryGbHours: Map[String, Double],

  /**
   * Per-phase peaks bucketed as samples arrive.
   * Heap is gap-free (JVM resets peak per tick), while direct memory is sampled
   * and might miss sub-tick spikes.
   */
  phaseWindowPeakExecutorMemoryBytes: Map[String, Long],
  phaseWindowTotalExecutorMemoryBytes: Map[String, Long],
  phaseWindowPeakDirectMemoryBytes: Map[String, Long],
  phaseWindowTotalDirectMemoryBytes: Map[String, Long],
  phaseWindowDriverPeakHeapBytes: Map[String, Long],
  phaseWindowDriverPeakDirectMemoryBytes: Map[String, Long],

  // Note: I/O volume is deliberately omitted. Spark (compressed storage bytes) and
  // Flink (serialized stream bytes) metrics are fundamentally incomparable.

  // Whole-JVM process metrics (Executors)
  jvmGcTimeMs: Long,
  executorCpuTimeNs: Long,
  avgCpuCoresBusy: Option[Double],
  cpuCoreHours: Double,

  peakExecutorMemoryBytes: Long,
  totalExecutorMemoryBytes: Long,
  memoryGbHours: Double,

  peakDirectMemoryBytes: Long,
  totalDirectMemoryBytes: Long,

  // Driver JVM metrics (tracked separately as driver-local work is invisible to executors)
  driverCpuTimeNs: Long,
  driverCpuCoreHours: Double,
  driverPeakHeapBytes: Long,
  driverGcTimeMs: Long,
  driverMemoryGbHours: Double,
  driverPeakDirectMemoryBytes: Long,

  // Evaluation metrics
  nClusters: Option[Int],
  noiseFraction: Option[Double],
  silhouette: Option[Double],

  /**
   * Silhouette internals.
   * `silhouetteSampleClusters` shows the actual number of clusters drawn in the sample.
   * If lower than `nClusters`, some clusters were missed entirely.
   */
  silhouetteScoredPoints: Option[Int],
  silhouetteSampleClusters: Option[Int],
  silhouetteUnscoredPoints: Option[Int],
  clusterSizes: Option[Map[String, String]],

  /**
   * Centroid-based indices computed on the FULL dataset.
   * DB: lower-is-better (bounded below by 0).
   * CH: higher-is-better (unbounded, comparable only within the same dataset).
   */
  daviesBouldin: Option[Double],
  calinskiHarabasz: Option[Double]
)

object RunResult {

  private implicit val formats: Formats = DefaultFormats

  final case class Timings(loadMs: Long, fitMs: Long, evalMs: Long, totalMs: Long)

  final case class PhaseMetrics(
    durationsMs: Map[String, Long],
    cpuTimeNs: Map[String, Long],
    gcTimeMs: Map[String, Long],
    memoryGbHours: Map[String, Double],
    driverCpuTimeNs: Map[String, Long],
    driverGcTimeMs: Map[String, Long],
    driverMemoryGbHours: Map[String, Double],
    windowPeakWorkerHeapBytes: Map[String, Long],
    windowTotalWorkerHeapBytes: Map[String, Long],
    windowPeakWorkerDirectBytes: Map[String, Long],
    windowTotalWorkerDirectBytes: Map[String, Long],
    windowDriverHeapBytes: Map[String, Long],
    windowDriverDirectBytes: Map[String, Long]
  )

  object PhaseMetrics {
    val empty: PhaseMetrics = PhaseMetrics(
      Map.empty, Map.empty, Map.empty, Map.empty, Map.empty, Map.empty, Map.empty,
      Map.empty, Map.empty, Map.empty, Map.empty, Map.empty, Map.empty
    )

    def from(
      deltas: Seq[(String, ProcessCpuPlugin.PhaseDelta)],
      durationsMs: Map[String, Long],
      peaks: Seq[(String, ProcessCpuPlugin.PhasePeaks)]
    ): PhaseMetrics = PhaseMetrics(
      durationsMs = durationsMs,
      cpuTimeNs = deltas.map { case (name, delta) => name -> delta.cpuNanos }.toMap,
      gcTimeMs = deltas.map { case (name, delta) => name -> delta.gcTimeMs }.toMap,
      memoryGbHours = deltas.map { case (name, delta) => name -> delta.heapByteSeconds / ByteSecondsPerGbHour }.toMap,
      driverCpuTimeNs = deltas.map { case (name, delta) => name -> delta.driverCpuNanos }.toMap,
      driverGcTimeMs = deltas.map { case (name, delta) => name -> delta.driverGcTimeMs }.toMap,
      driverMemoryGbHours = deltas.map { case (name, delta) => name -> delta.driverHeapByteSeconds / ByteSecondsPerGbHour }.toMap,
      windowPeakWorkerHeapBytes = peaks.map { case (name, peak) => name -> peak.maxWorkerHeapBytes }.toMap,
      windowTotalWorkerHeapBytes = peaks.map { case (name, peak) => name -> peak.totalWorkerHeapBytes }.toMap,
      windowPeakWorkerDirectBytes = peaks.map { case (name, peak) => name -> peak.maxWorkerDirectBytes }.toMap,
      windowTotalWorkerDirectBytes = peaks.map { case (name, peak) => name -> peak.totalWorkerDirectBytes }.toMap,
      windowDriverHeapBytes = peaks.map { case (name, peak) => name -> peak.driverHeapBytes }.toMap,
      windowDriverDirectBytes = peaks.map { case (name, peak) => name -> peak.driverDirectBytes }.toMap
    )
  }

  final case class ProcessMetrics(
    cpuNanos: Long,
    maxWorkerHeapBytes: Long,
    totalHeapBytes: Long,
    heapByteSeconds: Double,
    gcTimeMs: Long,
    maxWorkerDirectBytes: Long,
    totalDirectBytes: Long,
    driverCpuNanos: Long,
    driverPeakHeapBytes: Long,
    driverHeapByteSeconds: Double,
    driverGcTimeMs: Long,
    driverPeakDirectBytes: Long
  )

  private val NanosPerCoreHour: Double = 3.6e12
  private val ByteSecondsPerGbHour: Double = 1024.0 * 1024 * 1024 * 3600.0

  final case class Workload(
    dataset: String,
    datasetMetadata: Map[String, String],
    nRows: Long,
    nPartitions: Int,
    nFeatures: Option[Int]
  )

  object Workload {
    def empty(config: RunConfig): Workload =
      Workload(config.dataset.`type`, Map.empty, nRows = -1L, nPartitions = -1, nFeatures = None)
  }

  def from(
    config: RunConfig,
    framework: String,
    profile: String,
    startedAtIso: String,
    finishedAtIso: String,
    status: String,
    errorMessage: Option[String],
    workload: Workload,
    timings: Timings,
    phases: PhaseMetrics,
    process: ProcessMetrics,
    eval: EvaluationResult
  ): RunResult = RunResult(
    runId = config.runId,
    framework = framework,
    profile = profile,
    startedAtIso = startedAtIso,
    finishedAtIso = finishedAtIso,
    status = status,
    errorMessage = errorMessage,
    algorithm = config.algorithm.name,
    algorithmParams = stringifyParams(config.algorithm.params),
    dataset = workload.dataset,
    datasetMetadata = workload.datasetMetadata,
    sparkConf = config.spark_config,
    experimentMetadata = config.experimentMetadata,
    nRows = workload.nRows,
    nPartitions = workload.nPartitions,
    nFeatures = workload.nFeatures,
    loadDurationMs = timings.loadMs,
    fitDurationMs = timings.fitMs,
    evalDurationMs = timings.evalMs,
    totalDurationMs = timings.totalMs,
    phaseDurationsMs = phases.durationsMs,
    phaseCpuTimeNs = phases.cpuTimeNs,
    phaseGcTimeMs = phases.gcTimeMs,
    phaseMemoryGbHours = phases.memoryGbHours,
    phaseDriverCpuTimeNs = phases.driverCpuTimeNs,
    phaseDriverGcTimeMs = phases.driverGcTimeMs,
    phaseDriverMemoryGbHours = phases.driverMemoryGbHours,
    phaseWindowPeakExecutorMemoryBytes = phases.windowPeakWorkerHeapBytes,
    phaseWindowTotalExecutorMemoryBytes = phases.windowTotalWorkerHeapBytes,
    phaseWindowPeakDirectMemoryBytes = phases.windowPeakWorkerDirectBytes,
    phaseWindowTotalDirectMemoryBytes = phases.windowTotalWorkerDirectBytes,
    phaseWindowDriverPeakHeapBytes = phases.windowDriverHeapBytes,
    phaseWindowDriverPeakDirectMemoryBytes = phases.windowDriverDirectBytes,
    jvmGcTimeMs = process.gcTimeMs,
    executorCpuTimeNs = process.cpuNanos,
    avgCpuCoresBusy = calculateAvgCpuCoresBusy(process.cpuNanos, timings.totalMs),
    cpuCoreHours = process.cpuNanos / NanosPerCoreHour,
    peakExecutorMemoryBytes = process.maxWorkerHeapBytes,
    totalExecutorMemoryBytes = process.totalHeapBytes,
    memoryGbHours = process.heapByteSeconds / ByteSecondsPerGbHour,
    peakDirectMemoryBytes = process.maxWorkerDirectBytes,
    totalDirectMemoryBytes = process.totalDirectBytes,
    driverCpuTimeNs = process.driverCpuNanos,
    driverCpuCoreHours = process.driverCpuNanos / NanosPerCoreHour,
    driverPeakHeapBytes = process.driverPeakHeapBytes,
    driverGcTimeMs = process.driverGcTimeMs,
    driverMemoryGbHours = process.driverHeapByteSeconds / ByteSecondsPerGbHour,
    driverPeakDirectMemoryBytes = process.driverPeakDirectBytes,
    nClusters = eval.nClusters,
    noiseFraction = eval.noiseFraction,
    silhouette = eval.silhouette,
    silhouetteScoredPoints = eval.silhouetteScoredPoints,
    silhouetteSampleClusters = eval.silhouetteSampleClusters,
    silhouetteUnscoredPoints = eval.silhouetteUnscoredPoints,
    clusterSizes = eval.clusterSizes.map(_.map { case (k, v) => k.toString -> v.toString }),
    daviesBouldin = eval.daviesBouldin,
    calinskiHarabasz = eval.calinskiHarabasz
  )

  private def calculateAvgCpuCoresBusy(cpuNanos: Long, totalDurationMs: Long): Option[Double] =
    if (totalDurationMs <= 0) None
    else Some(cpuNanos / (totalDurationMs.toDouble * 1e6))

  private def stringifyParams(params: JObject): Map[String, String] =
    params.obj.map { case (key, value) =>
      key -> (value match {
        case JString(s) => s
        case JBool(b) => b.toString
        case JInt(n) => n.toString
        case JLong(n) => n.toString
        case JDouble(d) => d.toString
        case JDecimal(d) => d.toString
        case JNull => "null"
        case other => compact(render(other))
      })
    }.toMap

  def toJsonString(runResult: RunResult): String =
    compact(render(Extraction.decompose(runResult)))

  /**
   * Writes the result to `<outputDir>/<runId>.json`.
   * Uses an atomic move from a temporary file to avoid partial reads by downstream systems.
   */
  def writeToDir(runResult: RunResult, outputDir: String): Path = {
    val outputDirectory = Paths.get(outputDir)
    Files.createDirectories(outputDirectory)

    val targetFile = outputDirectory.resolve(s"${runResult.runId}.json")
    val tempFile = outputDirectory.resolve(s".${runResult.runId}.json.tmp")

    Files.write(
      tempFile,
      toJsonString(runResult).getBytes(StandardCharsets.UTF_8),
      StandardOpenOption.CREATE,
      StandardOpenOption.TRUNCATE_EXISTING,
      StandardOpenOption.WRITE
    )

    try {
      Files.move(tempFile, targetFile, StandardCopyOption.ATOMIC_MOVE)
    } catch {
      case _: AtomicMoveNotSupportedException =>
        Files.move(tempFile, targetFile, StandardCopyOption.REPLACE_EXISTING)
    }

    targetFile
  }
}
