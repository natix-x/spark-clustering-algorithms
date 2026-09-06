package clustering.benchmark.metrics

import com.sun.management.OperatingSystemMXBean
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.api.plugin.{DriverPlugin, ExecutorPlugin, PluginContext, SparkPlugin}

import org.log4s.getLogger

import java.lang.management.{BufferPoolMXBean, ManagementFactory, MemoryType}
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.{Executors, ScheduledExecutorService, TimeUnit}
import scala.jdk.CollectionConverters._
import scala.util.control.NonFatal

/**
 * Single tick snapshot of JVM resources.
 *
 * - cpuNanos, heapByteSeconds, gcTimeMs are monotonic/cumulative.
 * - tickPeakHeapBytes is a true gap-free peak for the last window (read-and-reset).
 * - directBytes is a point-in-time sample.
 */
private[metrics] final case class JvmSample(cpuNanos: Long,
                                           tickPeakHeapBytes: Long,
                                           heapByteSeconds: Double,
                                           gcTimeMs: Long,
                                           directBytes: Long
                                           )

private[metrics] final case class WorkerSample(execId: String, sample: JvmSample)

/**
 * Gathers JVM metrics from executors and the driver via Spark Plugins.
 * Designed as a global sink for a single benchmark run.
 */
object ProcessCpuPlugin {
  // Single map keyed by execId: each executor's fields are merged atomically together,
  // so a concurrent snapshot() read can never tear one executor's sample across fields.
  private val executorSamples = new ConcurrentHashMap[String, JvmSample]()

  private val phaseWindowHeapMap = new ConcurrentHashMap[String, java.lang.Long]()
  private val phaseWindowDirectMap = new ConcurrentHashMap[String, java.lang.Long]()

  val SamplingIntervalConf = "spark.clustering.metrics.samplingIntervalMs"
  val DefaultSamplingIntervalMs = 200L // 200 milliseconds

  // How often an executor flushes its locally-aggregated JvmSample to the driver via RPC.
  // Decoupled from SamplingIntervalConf: sampling stays tick-granular (gap-free per-tick peaks),
  // only the network send is throttled, so peak precision is unaffected by send frequency.
  val SendIntervalConf = "spark.clustering.metrics.sendIntervalMs"
  val DefaultSendIntervalMs = 2000L // 2 seconds

  def samplingIntervalMs(conf: SparkConf): Long =
    conf.getOption(SamplingIntervalConf).map(_.toLong).filter(_ > 0).getOrElse(DefaultSamplingIntervalMs)

  def sendIntervalMs(conf: SparkConf): Long =
    conf.getOption(SendIntervalConf).map(_.toLong).filter(_ > 0).getOrElse(DefaultSendIntervalMs)

  def settleMs(samplingIntervalMs: Long): Long =
    math.max(600L, 3L * samplingIntervalMs)

  final case class CounterSnapshot(
                                    executorCpuNanos: Map[String, Long],
                                    executorGcTimeMs: Map[String, Long],
                                    executorHeapByteSeconds: Map[String, Double],
                                    driverCpuNanos: Long,
                                    driverGcTimeMs: Long,
                                    driverHeapByteSeconds: Double
                                  )

  object CounterSnapshot {
    val empty: CounterSnapshot = CounterSnapshot(Map.empty, Map.empty, Map.empty, 0L, 0L, 0.0)
  }

  final case class PhaseDelta(cpuNanos: Long,
                              gcTimeMs: Long,
                              heapByteSeconds: Double,
                              driverCpuNanos: Long,
                              driverGcTimeMs: Long,
                              driverHeapByteSeconds: Double
                             )

  /**
   * Retrieves a point-in-time copy of all cumulative counters.
   */
  def snapshot(): CounterSnapshot = synchronized {
    val samples = executorSamples.asScala.toMap
    CounterSnapshot(
      executorCpuNanos = samples.map { case (id, s) => id -> s.cpuNanos },
      executorGcTimeMs = samples.map { case (id, s) => id -> s.gcTimeMs },
      executorHeapByteSeconds = samples.map { case (id, s) => id -> s.heapByteSeconds },
      driverCpuNanos = driverCpuNanos,
      driverGcTimeMs = driverGcTimeMs,
      driverHeapByteSeconds = driverHeapByteSeconds
    )
  }

  def between(from: CounterSnapshot, to: CounterSnapshot): PhaseDelta = PhaseDelta(
    cpuNanos = calculateDeltaSum(to.executorCpuNanos, from.executorCpuNanos),
    gcTimeMs = calculateDeltaSum(to.executorGcTimeMs, from.executorGcTimeMs),
    heapByteSeconds = calculateDeltaSumDouble(to.executorHeapByteSeconds, from.executorHeapByteSeconds),
    driverCpuNanos = math.max(0L, to.driverCpuNanos - from.driverCpuNanos),
    driverGcTimeMs = math.max(0L, to.driverGcTimeMs - from.driverGcTimeMs),
    driverHeapByteSeconds = math.max(0.0, to.driverHeapByteSeconds - from.driverHeapByteSeconds)
  )

  private def calculateDeltaSum(to: Map[String, Long], from: Map[String, Long]): Long =
    to.foldLeft(0L) { case (acc, (id, value)) => acc + math.max(0L, value - from.getOrElse(id, 0L)) }

  private def calculateDeltaSumDouble(to: Map[String, Double], from: Map[String, Double]): Double =
    to.foldLeft(0.0) { case (acc, (id, value)) => acc + math.max(0.0, value - from.getOrElse(id, 0.0)) }

  private var runBaseline: CounterSnapshot = CounterSnapshot.empty

  private[metrics] def report(execId: String, sample: JvmSample): Unit = {
    executorSamples.merge(execId, sample, { (old, incoming) =>
      JvmSample(
        cpuNanos = math.max(old.cpuNanos, incoming.cpuNanos),
        tickPeakHeapBytes = math.max(old.tickPeakHeapBytes, incoming.tickPeakHeapBytes),
        heapByteSeconds = math.max(old.heapByteSeconds, incoming.heapByteSeconds),
        gcTimeMs = math.max(old.gcTimeMs, incoming.gcTimeMs),
        directBytes = math.max(old.directBytes, incoming.directBytes)
      )
    })

    // Guarded by the same monitor as beginPhase()/endPhase(): a sample that arrives in the gap
    // between endPhase() and the next beginPhase() (isPhaseOpen == false) is dropped here instead
    // of sitting in the map until the next phase opens and silently inheriting it. Does not fix a
    // sample that arrives AFTER the next phase has already opened - that still needs a phase-id/
    // timestamp on WorkerSample, deliberately not built (see clock-skew note elsewhere).
    synchronized {
      if (isPhaseOpen) {
        phaseWindowHeapMap.merge(execId, sample.tickPeakHeapBytes, Math.max)
        phaseWindowDirectMap.merge(execId, sample.directBytes, Math.max)
      }
    }
  }

  final case class PhasePeaks(maxWorkerHeapBytes: Long,
                              totalWorkerHeapBytes: Long,
                              maxWorkerDirectBytes: Long,
                              totalWorkerDirectBytes: Long,
                              driverHeapBytes: Long,
                              driverDirectBytes: Long
                             )

  private var phaseDriverPeakHeapBytes: Long = 0L
  private var phaseDriverPeakDirectBytes: Long = 0L
  private var isPhaseOpen: Boolean = false
  private val sealedPhases = scala.collection.mutable.LinkedHashMap[String, PhasePeaks]()

  def beginPhase(): Unit = synchronized {
    phaseWindowHeapMap.clear()
    phaseWindowDirectMap.clear()
    phaseDriverPeakHeapBytes = 0L
    phaseDriverPeakDirectBytes = 0L
    isPhaseOpen = true
  }

  def endPhase(phaseName: String): Unit = synchronized {
    if (isPhaseOpen) {
      val workerHeaps = phaseWindowHeapMap.values().asScala.map(_.toLong)
      val workerDirects = phaseWindowDirectMap.values().asScala.map(_.toLong)

      sealedPhases(phaseName) = PhasePeaks(
        maxWorkerHeapBytes = if (workerHeaps.isEmpty) 0L else workerHeaps.max,
        totalWorkerHeapBytes = workerHeaps.sum,
        maxWorkerDirectBytes = if (workerDirects.isEmpty) 0L else workerDirects.max,
        totalWorkerDirectBytes = workerDirects.sum,
        driverHeapBytes = phaseDriverPeakHeapBytes,
        driverDirectBytes = phaseDriverPeakDirectBytes
      )
      isPhaseOpen = false
    }
  }

  def windowPeaks: Seq[(String, PhasePeaks)] = synchronized(sealedPhases.toSeq)

  def captureBaseline(): Unit = synchronized {
    runBaseline = snapshot()
  }

  def baseline: CounterSnapshot = synchronized(runBaseline)

  def totalCpuNanos(): Long = between(baseline, snapshot()).cpuNanos

  def maxWorkerHeapBytes(): Long = {
    val peaks = executorSamples.values().asScala.map(_.tickPeakHeapBytes)
    if (peaks.isEmpty) 0L else peaks.max
  }

  def totalHeapBytes(): Long =
    executorSamples.values().asScala.map(_.tickPeakHeapBytes).sum

  def totalHeapByteSeconds(): Double = between(baseline, snapshot()).heapByteSeconds

  def totalGcTimeMs(): Long = between(baseline, snapshot()).gcTimeMs

  def maxWorkerDirectBytes(): Long = {
    val directs = executorSamples.values().asScala.map(_.directBytes)
    if (directs.isEmpty) 0L else directs.max
  }

  def totalDirectBytes(): Long =
    executorSamples.values().asScala.map(_.directBytes).sum

  // ── Driver JVM Metrics ──
  private var driverCpuNanos: Long = 0L
  private var driverPeakHeapBytes: Long = 0L
  private var driverHeapByteSeconds: Double = 0.0
  private var driverGcTimeMs: Long = 0L
  private var driverPeakDirectBytes: Long = 0L

  private[metrics] def reportDriver(sample: JvmSample): Unit = synchronized {
    driverCpuNanos = math.max(driverCpuNanos, sample.cpuNanos)
    driverPeakHeapBytes = math.max(driverPeakHeapBytes, sample.tickPeakHeapBytes)
    driverHeapByteSeconds = math.max(driverHeapByteSeconds, sample.heapByteSeconds)
    driverGcTimeMs = math.max(driverGcTimeMs, sample.gcTimeMs)
    driverPeakDirectBytes = math.max(driverPeakDirectBytes, sample.directBytes)

    phaseDriverPeakHeapBytes = math.max(phaseDriverPeakHeapBytes, sample.tickPeakHeapBytes)
    phaseDriverPeakDirectBytes = math.max(phaseDriverPeakDirectBytes, sample.directBytes)
  }

  def getDriverCpuNanos: Long = synchronized(math.max(0L, driverCpuNanos - runBaseline.driverCpuNanos))
  def getDriverPeakHeapBytes: Long = synchronized(driverPeakHeapBytes)
  def getDriverHeapByteSeconds: Double = synchronized(math.max(0.0, driverHeapByteSeconds - runBaseline.driverHeapByteSeconds))
  def getDriverGcTimeMs: Long = synchronized(math.max(0L, driverGcTimeMs - runBaseline.driverGcTimeMs))
  def getDriverPeakDirectBytes: Long = synchronized(driverPeakDirectBytes)

  def reset(): Unit = {
    executorSamples.clear()
    phaseWindowHeapMap.clear()
    phaseWindowDirectMap.clear()

    synchronized {
      phaseDriverPeakHeapBytes = 0L
      phaseDriverPeakDirectBytes = 0L
      isPhaseOpen = false
      sealedPhases.clear()

      driverCpuNanos = 0L
      driverPeakHeapBytes = 0L
      driverHeapByteSeconds = 0.0
      driverGcTimeMs = 0L
      driverPeakDirectBytes = 0L
      runBaseline = CounterSnapshot.empty
    }
  }
}

/**
 * Periodically samples whole-JVM CPU, Heap, GC, and Direct memory.
 * Reads and immediately resets the JVM's peak heap usage to accurately capture per-tick peaks.
 */
private[metrics] final class JvmProcessSampler(threadName: String, intervalMs: Long)(publish: JvmSample => Unit) {

  private val logger = getLogger

  private val osBean = ManagementFactory.getPlatformMXBean(classOf[OperatingSystemMXBean])
  private val initialCpuNanos = osBean.getProcessCpuTime

  private val gcBeans = ManagementFactory.getGarbageCollectorMXBeans.asScala
  private val initialGcTimeMs = calculateTotalGcMs()

  private var scheduler: ScheduledExecutorService = _
  private var lastSampleNanos: Long = System.nanoTime()
  private var cumulativeHeapByteSeconds: Double = 0.0

  private val heapPools: Seq[java.lang.management.MemoryPoolMXBean] =
    ManagementFactory.getMemoryPoolMXBeans.asScala.filter(_.getType == MemoryType.HEAP).toSeq

  resetHeapPeaks()

  def start(): Unit = {
    scheduler = Executors.newSingleThreadScheduledExecutor { runnable =>
      val thread = new Thread(runnable, threadName)
      thread.setDaemon(true)
      thread
    }
    scheduler.scheduleAtFixedRate(() => sample(), intervalMs, intervalMs, TimeUnit.MILLISECONDS)
  }

  def sample(): Unit = synchronized {
    try {
      val currentCpuNanos = osBean.getProcessCpuTime
      val cpuDelta = if (currentCpuNanos > 0L) math.max(0L, currentCpuNanos - initialCpuNanos) else 0L
      val gcDelta = math.max(0L, calculateTotalGcMs() - initialGcTimeMs)

      val now = System.nanoTime()
      val deltaSeconds = math.max(0L, now - lastSampleNanos) / 1e9
      lastSampleNanos = now

      cumulativeHeapByteSeconds += calculateCurrentHeapUsed() * deltaSeconds
      val tickPeakHeap = readAndResetHeapPeak()
      val directNow = calculateCurrentDirectUsed()

      publish(JvmSample(cpuDelta, tickPeakHeap, cumulativeHeapByteSeconds, gcDelta, directNow))
    } catch {
      case NonFatal(e) =>
        // A scheduleAtFixedRate task that throws is never retried - log so the sampler's
        // silence is visible instead of the run just going quiet.
        logger.error(e)(s"$threadName: sample tick failed, skipping this tick")
    }
  }

  def stop(): Unit = {
    try sample()
    finally if (scheduler != null) scheduler.shutdownNow()
  }

  /**
   * Reads the JVM's tracked peak usage and immediately resets it.
   * This guarantees that the next tick captures the peak specific to its own window.
   */
  private def readAndResetHeapPeak(): Long =
    heapPools.foldLeft(0L) { (totalPeak, pool) =>
      val peak = Option(pool.getPeakUsage).map(_.getUsed).getOrElse(0L)
      try pool.resetPeakUsage() catch { case _: UnsupportedOperationException => () }
      totalPeak + peak
    }

  private def resetHeapPeaks(): Unit =
    heapPools.foreach { pool =>
      try pool.resetPeakUsage() catch { case _: UnsupportedOperationException => () }
    }

  private def calculateCurrentHeapUsed(): Long =
    heapPools.flatMap(pool => Option(pool.getUsage).map(_.getUsed)).sum

  private def calculateCurrentDirectUsed(): Long =
    ManagementFactory.getPlatformMXBeans(classOf[BufferPoolMXBean]).asScala
      .filter(_.getName == "direct")
      .map(_.getMemoryUsed)
      .sum

  private def calculateTotalGcMs(): Long =
    gcBeans.map(bean => math.max(0L, bean.getCollectionTime)).sum
}

final class ProcessCpuPlugin extends SparkPlugin {
  override def driverPlugin(): DriverPlugin = new ProcessCpuDriverPlugin
  override def executorPlugin(): ExecutorPlugin = new ProcessCpuExecutorPlugin
}

final class ProcessCpuDriverPlugin extends DriverPlugin {
  private var sampler: JvmProcessSampler = _

  override def init(sc: SparkContext, ctx: PluginContext): java.util.Map[String, String] = {
    val intervalMs = ProcessCpuPlugin.samplingIntervalMs(sc.getConf)
    val sendMs = ProcessCpuPlugin.sendIntervalMs(sc.getConf)
    sampler = new JvmProcessSampler("driver-resource-sampler", intervalMs)(ProcessCpuPlugin.reportDriver)
    sampler.start()

    val conf = new java.util.HashMap[String, String]()
    conf.put(ProcessCpuPlugin.SamplingIntervalConf, intervalMs.toString)
    conf.put(ProcessCpuPlugin.SendIntervalConf, sendMs.toString)
    conf
  }

  override def receive(message: Any): AnyRef = message match {
    case WorkerSample(execId, sample) =>
      ProcessCpuPlugin.report(execId, sample)
      null
    case _ => null
  }

  override def shutdown(): Unit = if (sampler != null) sampler.stop()
}

final class ProcessCpuExecutorPlugin extends ExecutorPlugin {
  private val logger = getLogger

  private var sampler: JvmProcessSampler = _
  private var sendScheduler: ScheduledExecutorService = _
  private var ctx: PluginContext = _

  // Accumulates ticks between sends. cpuNanos/heapByteSeconds/gcTimeMs are cumulative-since-init
  // (monotonic, so "latest tick" already IS the running total - no reset needed). tickPeakHeapBytes
  // and directBytes are per-tick values (the former reset to 0 every tick by JvmProcessSampler, the
  // latter a point-in-time sample), so they are max-merged here to preserve a gap-free peak across
  // the whole send window, then reset to 0 once flushed so the next window starts clean.
  private val windowState = new java.util.concurrent.atomic.AtomicReference[JvmSample](
    JvmSample(0L, 0L, 0.0, 0L, 0L)
  )

  override def init(pluginCtx: PluginContext, extraConf: java.util.Map[String, String]): Unit = {
    ctx = pluginCtx
    val intervalMs = Option(extraConf.get(ProcessCpuPlugin.SamplingIntervalConf))
      .map(_.toLong).filter(_ > 0).getOrElse(ProcessCpuPlugin.DefaultSamplingIntervalMs)
    val sendMs = Option(extraConf.get(ProcessCpuPlugin.SendIntervalConf))
      .map(_.toLong).filter(_ > 0).getOrElse(ProcessCpuPlugin.DefaultSendIntervalMs)

    sampler = new JvmProcessSampler("worker-resource-reporter", intervalMs)(sample =>
      windowState.updateAndGet { merged =>
        JvmSample(
          cpuNanos = sample.cpuNanos,
          tickPeakHeapBytes = math.max(merged.tickPeakHeapBytes, sample.tickPeakHeapBytes),
          heapByteSeconds = sample.heapByteSeconds,
          gcTimeMs = sample.gcTimeMs,
          directBytes = math.max(merged.directBytes, sample.directBytes)
        )
      }
    )
    sampler.start()

    sendScheduler = Executors.newSingleThreadScheduledExecutor { runnable =>
      val thread = new Thread(runnable, "worker-resource-sender")
      thread.setDaemon(true)
      thread
    }
    sendScheduler.scheduleAtFixedRate(() => flush(), sendMs, sendMs, TimeUnit.MILLISECONDS)
  }

  private def flush(): Unit = {
    // Only the window-scoped peak fields reset - cpuNanos/heapByteSeconds/gcTimeMs are left as the
    // latest cumulative reading, since the very next tick overwrites them with a larger value anyway
    // and the driver's own report() merge is already Math.max-based, so a send never turns them back.
    val toSend = windowState.getAndUpdate(s => s.copy(tickPeakHeapBytes = 0L, directBytes = 0L))
    try ctx.send(WorkerSample(ctx.executorID(), toSend))
    catch { case NonFatal(e) => logger.warn(e)("failed to send worker resource sample to driver") }
  }

  override def shutdown(): Unit = {
    if (sampler != null) sampler.stop()
    if (sendScheduler != null) sendScheduler.shutdownNow()
    if (ctx != null) flush() // final flush so the last window isn't lost
  }
}
