package clustering.benchmark.metrics

import org.json4s._
import org.json4s.jackson.JsonMethods._

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths, StandardOpenOption}

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

  // Spark listener — spill (> 0 oznacza za mało pamięci dla executora)
  diskBytesSpilled:        Long,
  memoryBytesSpilled:      Long,

  // Spark listener — CPU
  jvmGcTimeMs:             Long,
  executorCpuTimeNs:       Long,
  executorRunTimeMs:       Long,

  // Spark listener — sieć / shuffle timing
  shuffleFetchWaitTimeMs:  Long,
  shuffleWriteTimeNs:      Long,

  // Spark listener — taski / stagi / pamięć
  taskCount:               Long,
  failedTaskCount:         Long,
  stageCount:              Long,
  totalStageMs:            Long,
  peakExecutorMemoryBytes: Long,

  // evaluation
  nClusters:           Int,
  noiseFraction:       Double,
  silhouette:          Option[Double],
  clusterSizes:        Map[String, Long]         // keys are stringified cluster ids for JSON friendliness
) {
  /** CPU efficiency: jaka część czasu executor faktycznie liczył [0.0–1.0].
   *  Wartości < 0.3 sugerują bottleneck sieci lub I/O.
   *  None gdy executorRunTimeMs == 0 (np. run zakończony błędem przed wykonaniem tasków).
   */
  def cpuEfficiency: Option[Double] =
    if (executorRunTimeMs == 0L) None
    else Some((executorCpuTimeNs / 1e6) / executorRunTimeMs)
}

object RunResult {

  private implicit val formats: Formats = DefaultFormats

  /** Render a RunResult as a single-line compact JSON.
   *  Perfect for `pd.read_json(..., lines=True)` in Python/Pandas.
   */
  def toJsonString(r: RunResult): String =
    compact(render(Extraction.decompose(r)))

  /** Write the result to `<outputDir>/<runId>.json`, creating parents if
   *  needed. Atomic-ish: write to a tmp sibling then rename, so partial files
   *  never appear under the final name (matters with array jobs scraping
   *  results in parallel).
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

    Files.move(tmp, target, java.nio.file.StandardCopyOption.REPLACE_EXISTING)
    target
  }

  /** Pomocnicza metoda do przepisania pól z ListenerSnapshot na RunResult.
   *  Wywołuj ją przy budowaniu RunResult po zakończeniu joba:
   *
   *  {{{
   *    val snap = listener.snapshot()
   *    val result = RunResult(
   *      ...,
   *      RunResult.listenerFields(snap): _*   // NIE tak — patrz przykład niżej
   *    )
   *  }}}
   *
   *  Ponieważ Scala case class nie wspiera spread przy konstruktorze,
   *  użyj named parameters bezpośrednio:
   *
   *  {{{
   *    val snap = listener.snapshot()
   *    RunResult(
   *      runId            = ...,
   *      // ...inne pola...
   *      shuffleReadBytes        = snap.shuffleReadBytes,
   *      shuffleWriteBytes       = snap.shuffleWriteBytes,
   *      inputBytes              = snap.inputBytes,
   *      outputBytes             = snap.outputBytes,
   *      diskBytesSpilled        = snap.diskBytesSpilled,
   *      memoryBytesSpilled      = snap.memoryBytesSpilled,
   *      jvmGcTimeMs             = snap.jvmGcTimeMs,
   *      executorCpuTimeNs       = snap.executorCpuTimeNs,
   *      executorRunTimeMs       = snap.executorRunTimeMs,
   *      shuffleFetchWaitTimeMs  = snap.shuffleFetchWaitTimeMs,
   *      shuffleWriteTimeNs      = snap.shuffleWriteTimeNs,
   *      taskCount               = snap.taskCount,
   *      failedTaskCount         = snap.failedTaskCount,
   *      stageCount              = snap.stageCount,
   *      totalStageMs            = snap.totalStageMs,
   *      peakExecutorMemoryBytes = snap.peakExecutorMemoryBytes,
   *    )
   *  }}}
   */
  def fromSnapshot(snap: BenchmarkListener.ListenerSnapshot): ListenerFields =
    ListenerFields(
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
    )

  /** Pośrednia struktura ułatwiająca przekazanie pól listenera do RunResult
   *  bez powtarzania wszystkich 16 nazw w każdym miejscu gdzie budujesz wynik.
   */
  final case class ListenerFields(
    shuffleReadBytes:        Long,
    shuffleWriteBytes:       Long,
    inputBytes:              Long,
    outputBytes:             Long,
    diskBytesSpilled:        Long,
    memoryBytesSpilled:      Long,
    jvmGcTimeMs:             Long,
    executorCpuTimeNs:       Long,
    executorRunTimeMs:       Long,
    shuffleFetchWaitTimeMs:  Long,
    shuffleWriteTimeNs:      Long,
    taskCount:               Long,
    failedTaskCount:         Long,
    stageCount:              Long,
    totalStageMs:            Long,
    peakExecutorMemoryBytes: Long,
  )
}