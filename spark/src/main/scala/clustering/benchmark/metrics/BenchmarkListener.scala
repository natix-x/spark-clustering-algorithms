package clustering.benchmark.metrics

// TODO: investigate if we could add something more here, leaving it for now
import org.apache.spark.Success
import org.apache.spark.scheduler._

import java.util.concurrent.atomic.{AtomicLong, AtomicReference}

/** Accumulates execution metrics from a running Spark job.
 *
 *  Hooked to the driver via `sparkContext.addSparkListener` for the duration
 *  of one benchmark run; the result is read once at the end and merged into
 *  the `RunResult`. All counters are atomic so listener callbacks (executor
 *  threads) and the driver thread don't race.
 */
final class BenchmarkListener extends SparkListener {

  // ── I/O ───────────────────────────────────────────────────────────────────
  private val shuffleReadBytes  = new AtomicLong(0L)
  private val shuffleWriteBytes = new AtomicLong(0L)
  private val inputBytes        = new AtomicLong(0L)
  private val outputBytes       = new AtomicLong(0L)

  // ── Spill (za mała pamięć → Spark zapisuje na dysk) ───────────────────────
  private val diskBytesSpilled   = new AtomicLong(0L)
  private val memoryBytesSpilled = new AtomicLong(0L)

  // ── CPU ───────────────────────────────────────────────────────────────────
  private val jvmGcTimeMs     = new AtomicLong(0L)
  /** Czas CPU faktycznie zajętego obliczeniami [ns]. */
  private val executorCpuTime = new AtomicLong(0L)
  /** Całkowity czas wykonania tasków [ms] — mianownik dla CPU efficiency. */
  private val executorRunTime = new AtomicLong(0L)

  // ── Sieć / Shuffle timing ─────────────────────────────────────────────────
  /** Czas oczekiwania na dane z innych węzłów podczas shuffle fetch [ms]. */
  private val shuffleFetchWaitTime = new AtomicLong(0L)
  /** Czas zapisu shuffle [ns]. */
  private val shuffleWriteTime     = new AtomicLong(0L)

  // ── Taski i stagi ─────────────────────────────────────────────────────────
  private val taskCount       = new AtomicLong(0L)
  private val failedTaskCount = new AtomicLong(0L)
  private val stageCount      = new AtomicLong(0L)
  private val totalStageMs    = new AtomicLong(0L)

  // ── Unified memory: rozbicie execution/storage (spark-only) ───────────────
  // Peak per-executor wartości z ExecutorMetrics (Spark 3.0+), dostarczane przez
  // onStageExecutorMetrics. Trzymamy MAX po wszystkich executorach/stage'ach —
  // analogicznie do peakExecutorMemoryBytes (najcięższy worker).
  private val peakOnHeapExecution = new AtomicLong(0L)
  private val peakOnHeapStorage   = new AtomicLong(0L)
  private val peakOnHeapUnified   = new AtomicLong(0L)

  // Peak executor memory is collected by ProcessCpuPlugin (whole-JVM heap sampled
  // on each worker), not here: Spark delivers no executor metric updates in local
  // mode, and the plugin path matches Flink's Status.JVM.Memory.Heap.Used.

  // ── Czas joba ─────────────────────────────────────────────────────────────
  private val firstJobStart = new AtomicReference[Option[Long]](None)
  private val lastJobEnd    = new AtomicReference[Option[Long]](None)

  // ── Handlers ──────────────────────────────────────────────────────────────

  override def onTaskEnd(taskEnd: SparkListenerTaskEnd): Unit = {
    taskCount.incrementAndGet()
    if (taskEnd.reason != Success) failedTaskCount.incrementAndGet()

    val m = taskEnd.taskMetrics
    if (m == null) return

    jvmGcTimeMs.addAndGet(m.jvmGCTime)
    executorCpuTime.addAndGet(m.executorCpuTime)
    executorRunTime.addAndGet(m.executorRunTime)
    diskBytesSpilled.addAndGet(m.diskBytesSpilled)
    memoryBytesSpilled.addAndGet(m.memoryBytesSpilled)

    val sr = m.shuffleReadMetrics
    if (sr != null) {
      shuffleReadBytes.addAndGet(sr.totalBytesRead)
      shuffleFetchWaitTime.addAndGet(sr.fetchWaitTime)
    }

    val sw = m.shuffleWriteMetrics
    if (sw != null) {
      shuffleWriteBytes.addAndGet(sw.bytesWritten)
      shuffleWriteTime.addAndGet(sw.writeTime)
    }

    val ir = m.inputMetrics
    if (ir != null) inputBytes.addAndGet(ir.bytesRead)

    val ow = m.outputMetrics
    if (ow != null) outputBytes.addAndGet(ow.bytesWritten)
  }

  override def onStageCompleted(stageCompleted: SparkListenerStageCompleted): Unit = {
    stageCount.incrementAndGet()
    val info = stageCompleted.stageInfo
    for {
      submitted <- info.submissionTime
      completed <- info.completionTime
    } totalStageMs.addAndGet(completed - submitted)
  }

  override def onStageExecutorMetrics(m: SparkListenerStageExecutorMetrics): Unit = {
    def upd(a: AtomicLong, name: String): Unit = {
      val v = m.executorMetrics.getMetricValue(name)
      a.updateAndGet(prev => math.max(prev, v))
    }
    upd(peakOnHeapExecution, "OnHeapExecutionMemory")
    upd(peakOnHeapStorage,   "OnHeapStorageMemory")
    upd(peakOnHeapUnified,   "OnHeapUnifiedMemory")
  }

  override def onJobStart(jobStart: SparkListenerJobStart): Unit =
    firstJobStart.updateAndGet {
      case None    => Some(jobStart.time)
      case current => current
    }

  override def onJobEnd(jobEnd: SparkListenerJobEnd): Unit =
    lastJobEnd.set(Some(jobEnd.time))

  /** Snapshot of all collected metrics; safe to call after job completion. */
  def snapshot(): BenchmarkListener.ListenerSnapshot =
    BenchmarkListener.ListenerSnapshot(
      shuffleReadBytes        = shuffleReadBytes.get(),
      shuffleWriteBytes       = shuffleWriteBytes.get(),
      inputBytes              = inputBytes.get(),
      outputBytes             = outputBytes.get(),
      diskBytesSpilled        = diskBytesSpilled.get(),
      memoryBytesSpilled      = memoryBytesSpilled.get(),
      jvmGcTimeMs             = jvmGcTimeMs.get(),
      executorCpuTimeNs       = executorCpuTime.get(),
      executorRunTimeMs       = executorRunTime.get(),
      shuffleFetchWaitTimeMs  = shuffleFetchWaitTime.get(),
      shuffleWriteTimeNs      = shuffleWriteTime.get(),
      taskCount               = taskCount.get(),
      failedTaskCount         = failedTaskCount.get(),
      stageCount              = stageCount.get(),
      totalStageMs            = totalStageMs.get(),
      peakOnHeapExecutionBytes = peakOnHeapExecution.get(),
      peakOnHeapStorageBytes   = peakOnHeapStorage.get(),
      peakOnHeapUnifiedBytes   = peakOnHeapUnified.get(),
      firstJobStartMs         = firstJobStart.get(),
      lastJobEndMs            = lastJobEnd.get()
    )
}

object BenchmarkListener {

  final case class ListenerSnapshot(
    // I/O
    shuffleReadBytes:        Long,
    shuffleWriteBytes:       Long,
    inputBytes:              Long,
    outputBytes:             Long,
    // Spill
    diskBytesSpilled:        Long,
    memoryBytesSpilled:      Long,
    // CPU
    jvmGcTimeMs:             Long,
    executorCpuTimeNs:       Long,
    executorRunTimeMs:       Long,
    // Sieć
    shuffleFetchWaitTimeMs:  Long,
    shuffleWriteTimeNs:      Long,
    // Taski / stagi
    taskCount:               Long,
    failedTaskCount:         Long,
    stageCount:              Long,
    totalStageMs:            Long,
    // Unified memory (execution/storage split, spark-only)
    peakOnHeapExecutionBytes: Long,
    peakOnHeapStorageBytes:   Long,
    peakOnHeapUnifiedBytes:   Long,
    // Czas
    firstJobStartMs:         Option[Long],
    lastJobEndMs:            Option[Long]
  )
}