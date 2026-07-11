package clustering.benchmark.metrics

import clustering.benchmark.config.RunConfig
import clustering.benchmark.evaluation.EvaluationResult
import clustering.benchmark.metrics.BenchmarkListener.ListenerSnapshot
import org.json4s._
import org.json4s.jackson.JsonMethods._

import java.nio.charset.StandardCharsets
import java.nio.file.{AtomicMoveNotSupportedException, Files, Path, Paths, StandardCopyOption, StandardOpenOption}

/** One row of benchmark output. Flat by design so analysis in pandas is
 *  a single `pd.read_json` over the results directory.
 *
 *  Nested-shaped fields (algorithmParams, datasetMetadata, sparkConf,
 *  clusterSizes) are kept as maps; the merger script flattens them when
 *  producing the analysis Parquet.
 */
final case class RunResult(
  // identity
  runId:               String,
  framework:           String,
  profile:             String,
  startedAtIso:        String,
  finishedAtIso:       String,
  status:              String,                   // "ok" | "failed"
  errorMessage:        Option[String],

  // experimental factors
  algorithm:           String,
  algorithmParams:     Map[String, String],
  dataset:             String,
  datasetMetadata:     Map[String, String],
  sparkConf:           Map[String, String],
  experimentMetadata:  Map[String, String],

  // workload
  nRows:               Long,
  nPartitions:         Int,
  nFeatures:           Option[Int],

  // wall-clock timings (driver, ms)
  loadDurationMs:      Long,
  fitDurationMs:       Long,
  evalDurationMs:      Long,
  totalDurationMs:     Long,

  // Spark listener — I/O
  shuffleReadBytes:        Long,
  shuffleWriteBytes:       Long,
  inputBytes:              Long,
  outputBytes:             Long,

  // Spark listener — spill (> 0 means the executor ran short of memory)
  diskBytesSpilled:        Long,
  memoryBytesSpilled:      Long,

  // Spark listener — CPU
  jvmGcTimeMs:             Long,
  executorCpuTimeNs:       Long,
  executorRunTimeMs:       Long,
  avgCpuCoresBusy:         Option[Double],   // cpu / (totalDurationMs * 1e6) = avg cores busy; cross-comparable

  // Spark listener — network / shuffle timing
  shuffleFetchWaitTimeMs:  Long,
  shuffleWriteTimeNs:      Long,

  // Spark listener — tasks / stages / memory
  taskCount:               Long,
  failedTaskCount:         Long,
  stageCount:              Long,
  totalStageMs:            Long,
  peakExecutorMemoryBytes: Long,   // MAX heap of single worker
  totalExecutorMemoryBytes: Long,  // SUM of per-worker heap peaks (cluster footprint)

  // Spark listener — unified memory split (execution/storage), spark-only.
  // Peak per-executor (MAX across executors/stages) from ExecutorMetrics; 0 in
  // local mode without polling. No Flink equivalent — null/0 on the Flink side.
  peakOnHeapExecutionBytes: Long,  // peak execution region (shuffle/join/sort/agg)
  peakOnHeapStorageBytes:   Long,  // peak storage region (cache/broadcast)
  peakOnHeapUnifiedBytes:   Long,  // peak whole unified pool (execution + storage)

  // evaluation — each is present only when the run requested that metric
  // (EvaluationSpec.metrics); otherwise None, and json4s omits the key.
  nClusters:           Option[Int],
  noiseFraction:       Option[Double],
  silhouette:          Option[Double],
  clusterSizes:        Option[Map[String, Long]]  // keys are stringified cluster ids for JSON friendliness
)

object RunResult {

  private implicit val formats: Formats = DefaultFormats

  // ── Parts handed in by the caller to assemble a RunResult ──────────────────
  // The job samples each of these at the right moment and passes them to `from`;
  // keeping them grouped is what lets `from` stay a plain field-by-field copy.

  /** Driver wall-clock timings for one run, in ms. */
  final case class Timings(loadMs: Long, fitMs: Long, evalMs: Long, totalMs: Long)

  /** Process-level metrics sampled once at the end of a run (from ProcessCpuPlugin). */
  final case class ProcessMetrics(cpuNanos: Long, maxWorkerHeapBytes: Long, totalHeapBytes: Long)

  /** Dataset workload facts measured during a run. */
  final case class Workload(
    dataset:         String,
    datasetMetadata: Map[String, String],
    nRows:           Long,
    nPartitions:     Int,
    nFeatures:       Option[Int]
  )
  object Workload {
    /** Placeholder for a failed run — only the configured dataset type is known. */
    def empty(config: RunConfig): Workload =
      Workload(config.dataset.`type`, Map.empty, nRows = -1L, nPartitions = -1, nFeatures = None)
  }

  /** Assemble one result row from the parts collected during a run.
   *
   *  The single place the output contract is built. Pure — no Spark, no clock or
   *  metric reads; the caller ([[clustering.benchmark.SparkClusteringJob]]) samples
   *  everything and hands it here. `status`/`errorMessage` and the (possibly empty)
   *  `workload`/`eval` are the only difference between an ok and a failed run, so
   *  there is one builder rather than a factory hierarchy. */
  def from(
    config:        RunConfig,
    framework:     String,
    profile:       String,
    startedAtIso:  String,
    finishedAtIso: String,
    status:        String,
    errorMessage:  Option[String],
    workload:      Workload,
    timings:       Timings,
    snap:          ListenerSnapshot,
    process:       ProcessMetrics,
    eval:          EvaluationResult
  ): RunResult = RunResult(
    runId                    = config.runId,
    framework                = framework,
    profile                  = profile,
    startedAtIso             = startedAtIso,
    finishedAtIso            = finishedAtIso,
    status                   = status,
    errorMessage             = errorMessage,
    algorithm                = config.algorithm.name,
    algorithmParams          = stringifyParams(config.algorithm.params),
    dataset                  = workload.dataset,
    datasetMetadata          = workload.datasetMetadata,
    sparkConf                = config.spark_config,
    experimentMetadata       = config.experimentMetadata,
    nRows                    = workload.nRows,
    nPartitions              = workload.nPartitions,
    nFeatures                = workload.nFeatures,
    loadDurationMs           = timings.loadMs,
    fitDurationMs            = timings.fitMs,
    evalDurationMs           = timings.evalMs,
    totalDurationMs          = timings.totalMs,
    shuffleReadBytes         = snap.shuffleReadBytes,
    shuffleWriteBytes        = snap.shuffleWriteBytes,
    inputBytes               = snap.inputBytes,
    outputBytes              = snap.outputBytes,
    diskBytesSpilled         = snap.diskBytesSpilled,
    memoryBytesSpilled       = snap.memoryBytesSpilled,
    jvmGcTimeMs              = snap.jvmGcTimeMs,
    executorCpuTimeNs        = process.cpuNanos,   // process CPU (Flink-comparable)
    executorRunTimeMs        = snap.executorRunTimeMs,
    avgCpuCoresBusy          = avgCpuCoresBusy(process.cpuNanos, timings.totalMs),
    shuffleFetchWaitTimeMs   = snap.shuffleFetchWaitTimeMs,
    shuffleWriteTimeNs       = snap.shuffleWriteTimeNs,
    taskCount                = snap.taskCount,
    failedTaskCount          = snap.failedTaskCount,
    stageCount               = snap.stageCount,
    totalStageMs             = snap.totalStageMs,
    peakExecutorMemoryBytes  = process.maxWorkerHeapBytes,   // MAX single worker
    totalExecutorMemoryBytes = process.totalHeapBytes,       // SUM across workers
    peakOnHeapExecutionBytes = snap.peakOnHeapExecutionBytes,
    peakOnHeapStorageBytes   = snap.peakOnHeapStorageBytes,
    peakOnHeapUnifiedBytes   = snap.peakOnHeapUnifiedBytes,
    nClusters                = eval.nClusters,
    noiseFraction            = eval.noiseFraction,
    silhouette               = eval.silhouette,
    clusterSizes             = eval.clusterSizes.map(_.map { case (k, v) => k.toString -> v })
  )

  /** Average number of CPU cores busy over the run: `cpuNs / (totalDurationMs * 1e6)`
   *  (CPU-seconds per wall-second = effective CPU parallelism). Same formula as the
   *  Flink port; divide by allocated cores for a 0–1 utilization. None when duration <= 0. */
  private def avgCpuCoresBusy(cpuNs: Long, totalDurationMs: Long): Option[Double] =
    if (totalDurationMs <= 0) None
    else Some(cpuNs / (totalDurationMs.toDouble * 1e6))

  /** json4s JObject -> Map[String, String] for the flat result schema: every param
   *  value is rendered to its string form. */
  private def stringifyParams(params: JObject): Map[String, String] =
    params.obj.map { case (k, v) =>
      k -> (v match {
        case JString(s)  => s
        case JBool(b)    => b.toString
        case JInt(n)     => n.toString
        case JLong(n)    => n.toString
        case JDouble(d)  => d.toString
        case JDecimal(d) => d.toString
        case JNull       => "null"
        case other       => compact(render(other))
      })
    }.toMap

  /** Render a RunResult as a single-line compact JSON.
   *  Perfect for `pd.read_json(..., lines=True)` in Python/Pandas.
   */
  def toJsonString(r: RunResult): String =
    compact(render(Extraction.decompose(r)))

  /** Write the result to `<outputDir>/<runId>.json`, creating parents if
   *  needed. Write to a tmp sibling then rename, so partial files never appear
   *  under the final name (matters with array jobs scraping results in
   *  parallel). The rename is atomic when the filesystem supports it, with a
   *  REPLACE_EXISTING fallback for those that don't.
   */
  def writeToDir(r: RunResult, outputDir: String): Path = {
    val dir    = Paths.get(outputDir)
    Files.createDirectories(dir)

    val target = dir.resolve(s"${r.runId}.json")
    val tmp    = dir.resolve(s".${r.runId}.json.tmp")

    Files.write(
      tmp,
      toJsonString(r).getBytes(StandardCharsets.UTF_8),
      StandardOpenOption.CREATE,
      StandardOpenOption.TRUNCATE_EXISTING,
      StandardOpenOption.WRITE
    )

    try Files.move(tmp, target, StandardCopyOption.ATOMIC_MOVE)
    catch {
      case _: AtomicMoveNotSupportedException =>
        Files.move(tmp, target, StandardCopyOption.REPLACE_EXISTING)
    }
    target
  }
}
