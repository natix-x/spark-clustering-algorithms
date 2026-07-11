package clustering.benchmark.metrics

import org.apache.spark.Success
import org.apache.spark.scheduler._

import java.util.concurrent.atomic.AtomicLong

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

  // ── Spill (too little memory → Spark writes to disk) ──────────────────────
  private val diskBytesSpilled   = new AtomicLong(0L)
  private val memoryBytesSpilled = new AtomicLong(0L)

  // ── CPU ───────────────────────────────────────────────────────────────────
  private val jvmGcTimeMs     = new AtomicLong(0L)
  // Whole-JVM process CPU (Flink-comparable) comes from ProcessCpuPlugin, not the
  // per-task TaskMetrics.executorCpuTime — so we don't accumulate the latter here.
  /** Total task execution time [ms]. */
  private val executorRunTime = new AtomicLong(0L)

  // ── Network / Shuffle timing ──────────────────────────────────────────────
  /** Time spent waiting for data from other nodes during shuffle fetch [ms]. */
  private val shuffleFetchWaitTime = new AtomicLong(0L)
  /** Shuffle write time [ns]. */
  private val shuffleWriteTime     = new AtomicLong(0L)

  // ── Tasks and stages ──────────────────────────────────────────────────────
  private val taskCount       = new AtomicLong(0L)
  private val failedTaskCount = new AtomicLong(0L)
  private val stageCount      = new AtomicLong(0L)
  private val totalStageMs    = new AtomicLong(0L)

  // ── Unified memory: execution/storage split (spark-only) ──────────────────
  // Peak per-executor values from ExecutorMetrics (Spark 3.0+), delivered via
  // onStageExecutorMetrics. We keep the MAX across all executors/stages —
  // analogous to peakExecutorMemoryBytes (the heaviest worker).
  private val peakOnHeapExecution = new AtomicLong(0L)
  private val peakOnHeapStorage   = new AtomicLong(0L)
  private val peakOnHeapUnified   = new AtomicLong(0L)

  // Peak executor memory is collected by ProcessCpuPlugin (whole-JVM heap sampled
  // on each worker), not here: Spark delivers no executor metric updates in local
  // mode, and the plugin path matches Flink's Status.JVM.Memory.Heap.Used.

  // ── Handlers ──────────────────────────────────────────────────────────────

  override def onTaskEnd(taskEnd: SparkListenerTaskEnd): Unit = {
    taskCount.incrementAndGet()
    if (taskEnd.reason != Success) failedTaskCount.incrementAndGet()

    val m = taskEnd.taskMetrics
    if (m == null) return

    jvmGcTimeMs.addAndGet(m.jvmGCTime)
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
    def updatePeak(peak: AtomicLong, metricName: String): Unit = {
      val value = m.executorMetrics.getMetricValue(metricName)
      peak.updateAndGet(prev => math.max(prev, value))
    }
    updatePeak(peakOnHeapExecution, "OnHeapExecutionMemory")
    updatePeak(peakOnHeapStorage,   "OnHeapStorageMemory")
    updatePeak(peakOnHeapUnified,   "OnHeapUnifiedMemory")
  }

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
      executorRunTimeMs       = executorRunTime.get(),
      shuffleFetchWaitTimeMs  = shuffleFetchWaitTime.get(),
      shuffleWriteTimeNs      = shuffleWriteTime.get(),
      taskCount               = taskCount.get(),
      failedTaskCount         = failedTaskCount.get(),
      stageCount              = stageCount.get(),
      totalStageMs            = totalStageMs.get(),
      peakOnHeapExecutionBytes = peakOnHeapExecution.get(),
      peakOnHeapStorageBytes   = peakOnHeapStorage.get(),
      peakOnHeapUnifiedBytes   = peakOnHeapUnified.get()
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
    executorRunTimeMs:       Long,
    // Network
    shuffleFetchWaitTimeMs:  Long,
    shuffleWriteTimeNs:      Long,
    // Tasks / stages
    taskCount:               Long,
    failedTaskCount:         Long,
    stageCount:              Long,
    totalStageMs:            Long,
    // Unified memory (execution/storage split, spark-only)
    peakOnHeapExecutionBytes: Long,
    peakOnHeapStorageBytes:   Long,
    peakOnHeapUnifiedBytes:   Long
  )
}
