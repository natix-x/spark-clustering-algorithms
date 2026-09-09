package clustering.benchmark.metrics

import org.scalatest.BeforeAndAfterEach
import org.scalatest.funsuite.AnyFunSuite

/** The run window, not the executor JVM lifetime, is what the benchmark reports.
 *
 *  Executor samplers publish values cumulative since their own plugin init (= executor JVM
 *  boot), which happens long before `awaitExecutors` returns, so every reading also covers
 *  per-executor startup and slot-registration idle. That error grows with node count, i.e. it
 *  bends the CPU-efficiency-vs-nodes curve — and the Flink side already corrects it
 *  (`BenchmarkListener.captureBaseline`), so leaving Spark uncorrected biases exactly the
 *  cross-framework comparison the thesis is built on. These tests pin the correction, and
 *  specifically pin WHICH fields get it: cumulative yes, peaks no (Flink rebases no peak
 *  either, and subtracting a baseline from a maximum is meaningless).
 *
 *  No Spark session here on purpose — `ProcessCpuPlugin` is a plain process-global sink, so the
 *  RPC arrival pattern can be replayed by calling `report` directly. */
class ProcessCpuBaselineSpec extends AnyFunSuite with BeforeAndAfterEach {

  override def beforeEach(): Unit = ProcessCpuPlugin.reset()
  override def afterEach(): Unit  = ProcessCpuPlugin.reset()

  /** One executor tick. `cpu`/`gc`/`heapSeconds` are cumulative-since-init; `heapPeak` is the
   *  JVM's tracked high-water mark for THAT TICK ALONE (the sampler resets it after every read)
   *  and `direct` is that tick's current buffer-pool reading. Both feed the run-level maxima and
   *  the open phase window — one reading, two consumers, exactly as the real sampler publishes
   *  it. */
  private def tick(execId: String, cpu: Long, heapPeak: Long = 0L, heapSeconds: Double = 0.0,
                   gc: Long = 0L, direct: Long = 0L): Unit =
    ProcessCpuPlugin.report(execId, JvmSample(cpu, heapPeak, heapSeconds, gc, direct))

  private def driverTick(cpu: Long, heapPeak: Long = 0L, heapSeconds: Double = 0.0,
                         gc: Long = 0L, direct: Long = 0L): Unit =
    ProcessCpuPlugin.reportDriver(JvmSample(cpu, heapPeak, heapSeconds, gc, direct))

  test("reset alone does not rebase: a post-reset tick re-delivers the whole since-init total") {
    tick("0", cpu = 100L, gc = 7L, heapSeconds = 500.0)   // pre-run ramp-up
    ProcessCpuPlugin.reset()
    tick("0", cpu = 100L, gc = 7L, heapSeconds = 500.0)   // same cumulative value, next tick
    ProcessCpuPlugin.captureBaseline()

    // Nothing happened since the baseline, so the run's share is zero — this is the assertion
    // that failed before captureBaseline existed (it read the full 100/7/500 ramp-up).
    assert(ProcessCpuPlugin.totalCpuNanos() == 0L)
    assert(ProcessCpuPlugin.totalGcTimeMs() == 0L)
    assert(ProcessCpuPlugin.totalHeapByteSeconds() == 0.0)
  }

  test("cumulative executor counters are reported as deltas since the baseline") {
    tick("0", cpu = 100L, gc = 7L, heapSeconds = 500.0)
    tick("1", cpu = 80L, gc = 3L, heapSeconds = 400.0)
    ProcessCpuPlugin.captureBaseline()

    tick("0", cpu = 250L, gc = 20L, heapSeconds = 1500.0)
    tick("1", cpu = 180L, gc = 8L, heapSeconds = 900.0)

    assert(ProcessCpuPlugin.totalCpuNanos() == (250 - 100) + (180 - 80))
    assert(ProcessCpuPlugin.totalGcTimeMs() == (20 - 7) + (8 - 3))
    assert(ProcessCpuPlugin.totalHeapByteSeconds() == (1500.0 - 500.0) + (900.0 - 400.0))
  }

  test("an executor that registered after the baseline contributes its whole reading") {
    tick("0", cpu = 100L)
    ProcessCpuPlugin.captureBaseline()

    tick("0", cpu = 150L)
    tick("1", cpu = 60L)   // joined late: its since-init total IS its run contribution

    assert(ProcessCpuPlugin.totalCpuNanos() == 50L + 60L)
  }

  test("a restarted executor floors at zero instead of subtracting into the negative") {
    tick("0", cpu = 100L)
    ProcessCpuPlugin.captureBaseline()
    tick("0", cpu = 40L)   // fresh JVM, counter below the baseline

    assert(ProcessCpuPlugin.totalCpuNanos() == 0L)
  }

  test("peaks are NOT rebased — a maximum has no meaningful baseline (matches Flink)") {
    tick("0", cpu = 0L, heapPeak = 900L, direct = 40L)
    ProcessCpuPlugin.captureBaseline()
    tick("0", cpu = 0L, heapPeak = 1000L, direct = 60L)
    tick("1", cpu = 0L, heapPeak = 700L, direct = 30L)

    assert(ProcessCpuPlugin.maxWorkerHeapBytes() == 1000L)
    assert(ProcessCpuPlugin.totalHeapBytes() == 1000L + 700L)
    assert(ProcessCpuPlugin.maxWorkerDirectBytes() == 60L)
    assert(ProcessCpuPlugin.totalDirectBytes() == 60L + 30L)
  }

  test("driver counters are rebased too — its sampler starts at SparkSession creation") {
    driverTick(cpu = 500L, heapPeak = 900L, heapSeconds = 2000.0, gc = 40L, direct = 10L)
    ProcessCpuPlugin.captureBaseline()
    driverTick(cpu = 1200L, heapPeak = 1100L, heapSeconds = 5000.0, gc = 65L, direct = 25L)

    assert(ProcessCpuPlugin.getDriverCpuNanos == 700L)
    assert(ProcessCpuPlugin.getDriverGcTimeMs == 25L)
    assert(ProcessCpuPlugin.getDriverHeapByteSeconds == 3000.0)
    // Driver peaks, like executor peaks, stay absolute.
    assert(ProcessCpuPlugin.getDriverPeakHeapBytes == 1100L)
    assert(ProcessCpuPlugin.getDriverPeakDirectBytes == 25L)
  }

  test("reset clears the baseline as well, so a second run in the same JVM starts clean") {
    tick("0", cpu = 100L)
    ProcessCpuPlugin.captureBaseline()
    ProcessCpuPlugin.reset()

    tick("0", cpu = 30L)   // second run, fresh executor reusing id "0"
    assert(ProcessCpuPlugin.totalCpuNanos() == 30L)
  }

  // ── Per-phase windows ─────────────────────────────────────────────────────
  // Same subtraction as the baseline, just between two arbitrary instants — which is what makes
  // "CPU of the fit phase only" expressible at all.

  test("a snapshot is a value, not a view of the live maps") {
    tick("0", cpu = 100L)
    val taken = ProcessCpuPlugin.snapshot()
    tick("0", cpu = 900L)

    assert(ProcessCpuPlugin.between(taken, ProcessCpuPlugin.snapshot()).cpuNanos == 800L)
  }

  test("between two snapshots gives that phase's use, across executors and for the driver") {
    tick("0", cpu = 100L, gc = 5L, heapSeconds = 100.0)
    tick("1", cpu = 100L, gc = 5L, heapSeconds = 100.0)
    driverTick(cpu = 50L, heapSeconds = 10.0, gc = 1L)
    val fitStart = ProcessCpuPlugin.snapshot()

    tick("0", cpu = 400L, gc = 12L, heapSeconds = 700.0)
    tick("1", cpu = 300L, gc = 9L, heapSeconds = 500.0)
    driverTick(cpu = 90L, heapSeconds = 60.0, gc = 4L)
    val fitEnd = ProcessCpuPlugin.snapshot()

    val fit = ProcessCpuPlugin.between(fitStart, fitEnd)
    assert(fit.cpuNanos == (400 - 100) + (300 - 100))
    assert(fit.gcTimeMs == (12 - 5) + (9 - 5))
    assert(fit.heapByteSeconds == (700.0 - 100.0) + (500.0 - 100.0))
    // The driver legs matter on their own: a driver-local phase is invisible to every
    // executor-side counter, which is exactly where the two engines diverge.
    assert(fit.driverCpuNanos == 90L - 50L)
    assert(fit.driverGcTimeMs == 4L - 1L)
    assert(fit.driverHeapByteSeconds == 60.0 - 10.0)
  }

  test("consecutive phases tile the run window, so they sum to the run-level totals") {
    tick("0", cpu = 100L, gc = 5L, heapSeconds = 100.0)   // pre-run ramp-up
    ProcessCpuPlugin.captureBaseline()
    val open = ProcessCpuPlugin.baseline

    tick("0", cpu = 250L, gc = 8L, heapSeconds = 400.0)
    val afterLoad = ProcessCpuPlugin.snapshot()
    tick("0", cpu = 900L, gc = 20L, heapSeconds = 1600.0)
    val afterFit = ProcessCpuPlugin.snapshot()
    tick("0", cpu = 1000L, gc = 25L, heapSeconds = 1900.0)
    val afterEval = ProcessCpuPlugin.snapshot()

    val phases = Seq(
      ProcessCpuPlugin.between(open, afterLoad),
      ProcessCpuPlugin.between(afterLoad, afterFit),
      ProcessCpuPlugin.between(afterFit, afterEval))

    assert(phases.map(_.cpuNanos).sum == ProcessCpuPlugin.totalCpuNanos())
    assert(phases.map(_.gcTimeMs).sum == ProcessCpuPlugin.totalGcTimeMs())
    assert(phases.map(_.heapByteSeconds).sum == ProcessCpuPlugin.totalHeapByteSeconds())
    // and the fit phase is the one the thesis is after, not the run total
    assert(phases(1).cpuNanos == 900L - 250L)
  }

  // ── Per-phase peaks ───────────────────────────────────────────────────────
  // A peak cannot be recovered from two readings of a cumulative peak after the fact, so the
  // samplers publish PER-TICK peaks and the driver buckets them as they arrive: it opens a
  // window, every sample max-merges into it, the window is sealed under the phase name. These
  // tests pin that the windows are genuinely disjoint — the whole point is that load's 10 GB
  // must not leak into fit — and that the same readings still add up to the run-level peak.

  test("a phase window keeps the max of the ticks that arrived inside it") {
    ProcessCpuPlugin.beginPhase()
    tick("0", cpu = 0L, heapPeak = 4000L, direct = 100L)
    tick("0", cpu = 0L, heapPeak = 9000L, direct = 300L)
    tick("0", cpu = 0L, heapPeak = 5000L, direct = 150L)
    ProcessCpuPlugin.endPhase("load")

    val peaks = ProcessCpuPlugin.windowPeaks.toMap
    assert(peaks("load").maxWorkerHeapBytes == 9000L)
    assert(peaks("load").maxWorkerDirectBytes == 300L)
  }

  test("a high peak in one phase does NOT leak into the next — the case differencing gets wrong") {
    ProcessCpuPlugin.beginPhase()
    tick("0", cpu = 0L, heapPeak = 10000L)        // load spikes to 10 GB
    ProcessCpuPlugin.endPhase("load")

    ProcessCpuPlugin.beginPhase()
    tick("0", cpu = 0L, heapPeak = 6000L)         // fit never exceeds 6 GB
    ProcessCpuPlugin.endPhase("fit")

    val peaks = ProcessCpuPlugin.windowPeaks.toMap
    assert(peaks("load").maxWorkerHeapBytes == 10000L)
    // Differencing the JVM's own high-water mark would have reported 0 here (the record was
    // never beaten), which is exactly why the per-tick readings are bucketed instead.
    assert(peaks("fit").maxWorkerHeapBytes == 6000L)
  }

  test("worker peaks aggregate as MAX-of-workers and SUM-of-workers, like the run-level pair") {
    ProcessCpuPlugin.beginPhase()
    tick("0", cpu = 0L, heapPeak = 7000L, direct = 200L)
    tick("1", cpu = 0L, heapPeak = 3000L, direct = 50L)
    ProcessCpuPlugin.endPhase("fit")

    val fit = ProcessCpuPlugin.windowPeaks.toMap.apply("fit")
    assert(fit.maxWorkerHeapBytes == 7000L)
    assert(fit.totalWorkerHeapBytes == 7000L + 3000L)
    assert(fit.maxWorkerDirectBytes == 200L)
    assert(fit.totalWorkerDirectBytes == 200L + 50L)
  }

  test("the driver gets its own peak per phase, never folded into the workers'") {
    ProcessCpuPlugin.beginPhase()
    tick("0", cpu = 0L, heapPeak = 3000L)
    driverTick(cpu = 0L, heapPeak = 8000L, direct = 40L)
    ProcessCpuPlugin.endPhase("fit")

    val fit = ProcessCpuPlugin.windowPeaks.toMap.apply("fit")
    assert(fit.maxWorkerHeapBytes == 3000L)
    assert(fit.driverHeapBytes == 8000L)
    assert(fit.driverDirectBytes == 40L)
  }

  test("ticks arriving with no window open are dropped, not charged to the last phase") {
    ProcessCpuPlugin.beginPhase()
    tick("0", cpu = 0L, heapPeak = 1000L)
    ProcessCpuPlugin.endPhase("fit")

    tick("0", cpu = 0L, heapPeak = 99000L)        // between phases
    ProcessCpuPlugin.endPhase("fit")            // no window open -> no-op

    assert(ProcessCpuPlugin.windowPeaks.toMap.apply("fit").maxWorkerHeapBytes == 1000L)
  }

  test("the run-level peak is the max over the phase windows — the tick windows tile the run") {
    // The invariant the per-tick reset buys, and the reason the run-level and per-phase peaks
    // can finally sit on one axis: both are maxima over the SAME per-tick readings, so a phase
    // peak can never exceed the run peak, and the run peak is always some phase's peak.
    ProcessCpuPlugin.beginPhase()
    tick("0", cpu = 0L, heapPeak = 4000L)
    ProcessCpuPlugin.endPhase("load")

    ProcessCpuPlugin.beginPhase()
    tick("0", cpu = 0L, heapPeak = 11000L)     // the run's true peak, inside fit
    tick("0", cpu = 0L, heapPeak = 2000L)      // and a later, lower tick: no longer masked by it
    ProcessCpuPlugin.endPhase("fit")

    val perPhase = ProcessCpuPlugin.windowPeaks.map { case (_, p) => p.maxWorkerHeapBytes }
    assert(perPhase.max == ProcessCpuPlugin.maxWorkerHeapBytes())
    assert(ProcessCpuPlugin.maxWorkerHeapBytes() == 11000L)
  }

  test("a later, lower tick does not lower the run-level peak") {
    // Per-tick readings are NOT monotonic (that is the whole point), so the driver must keep a
    // running max rather than the last value it was sent.
    tick("0", cpu = 0L, heapPeak = 9000L)
    tick("0", cpu = 0L, heapPeak = 1000L)
    assert(ProcessCpuPlugin.maxWorkerHeapBytes() == 9000L)
  }

  test("the boundary settle is derived from the sampling period, so 200ms costs less waiting") {
    assert(ProcessCpuPlugin.DefaultSamplingIntervalMs == 200L)
    assert(ProcessCpuPlugin.settleMs(200L) == 600L)
    assert(ProcessCpuPlugin.settleMs(1000L) == 3000L)   // scales up with a coarser period
    assert(ProcessCpuPlugin.settleMs(50L) == 600L)      // floored, RPC latency still has to fit
  }
}
