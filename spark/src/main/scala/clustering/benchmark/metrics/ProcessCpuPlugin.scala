package clustering.benchmark.metrics

import com.sun.management.OperatingSystemMXBean
import org.apache.spark.api.plugin.{DriverPlugin, ExecutorPlugin, PluginContext, SparkPlugin}

import java.lang.management.{ManagementFactory, MemoryType}
import java.util.concurrent.{ConcurrentHashMap, Executors, ScheduledExecutorService, TimeUnit}
import scala.jdk.CollectionConverters._
import scala.util.control.NonFatal

/** RPC message: one executor's process-CPU delta [ns] (since plugin init) and its
 *  peak JVM heap-used [bytes] (since plugin init). Top-level (not nested) so it
 *  serializes cleanly over the plugin RPC channel. */
private[metrics] final case class WorkerSample(execId: String, cpuNanos: Long, peakHeapBytes: Long)

/** Cross-framework-comparable engine resource metrics. Spark's standard listener
 *  gives PER-TASK CPU (`TaskMetrics.executorCpuTime`) and delivers no executor
 *  memory updates in local mode — neither is comparable with Flink, where the only
 *  CPU/RAM signals are WHOLE-JVM gauges (`Status.JVM.CPU.Time`,
 *  `Status.JVM.Memory.Heap.Used`). This plugin samples the SAME JVM MXBeans on
 *  every worker JVM, mirroring Flink's reporter, without forking Spark.
 *
 *  Each executor reports to the driver its process CPU (`getProcessCpuTime`, a
 *  monotonic delta since init) and its peak heap used since init
 *  (`MemoryPoolMXBean.getPeakUsage`, which the JVM tracks continuously — no
 *  sampling gap, so a spike between two reports is still captured). The driver
 *  keeps the MAX per executor. Aggregates SUM across executors — matching the
 *  Flink side, which sums per-TaskManager values. In `local[*]` the single JVM is
 *  driver+executor (includes driver), exactly as Flink's local MiniCluster.
 *  Register via {@code spark.plugins=clustering.benchmark.metrics.ProcessCpuPlugin}.
 *
 *  Note on design: this object is a process-global mutable sink, which is a
 *  deliberate framework boundary — Spark instantiates the plugin by reflection
 *  and the driver receives worker samples over RPC, with no handle we could
 *  inject state into. It is therefore not an injectable collaborator. The
 *  contract is: exactly one benchmark run per JVM (or sequential runs guarded by
 *  `reset()`); concurrent runs in one JVM would share this sink and corrupt each
 *  other's metrics. */
object ProcessCpuPlugin {
  private val cpuByExecutor  = new ConcurrentHashMap[String, java.lang.Long]()
  private val heapByExecutor = new ConcurrentHashMap[String, java.lang.Long]()

  private[metrics] def report(execId: String, cpuNanos: Long, peakHeapBytes: Long): Unit = {
    cpuByExecutor.merge(execId, cpuNanos, (a, b) => Math.max(a, b))
    heapByExecutor.merge(execId, peakHeapBytes, (a, b) => Math.max(a, b))
  }

  /** Total process CPU [ns] across all executors seen this run. */
  def totalCpuNanos(): Long = cpuByExecutor.values().asScala.foldLeft(0L)(_ + _)

  /** Peak JVM heap-used [bytes] of the single heaviest executor (MAX). */
  def maxWorkerHeapBytes(): Long =
    if (heapByExecutor.isEmpty) 0L else heapByExecutor.values().asScala.map(_.toLong).max

  /** Sum of per-executor peak JVM heap-used [bytes] = total cluster footprint
   *  (SUM; matches Flink's sum-across-TMs). */
  def totalHeapBytes(): Long = heapByExecutor.values().asScala.foldLeft(0L)(_ + _)

  /** Clear accumulated values before a run. */
  def reset(): Unit = { cpuByExecutor.clear(); heapByExecutor.clear() }
}

final class ProcessCpuPlugin extends SparkPlugin {
  override def driverPlugin(): DriverPlugin     = new ProcessCpuDriverPlugin
  override def executorPlugin(): ExecutorPlugin = new ProcessCpuExecutorPlugin
}

final class ProcessCpuDriverPlugin extends DriverPlugin {
  override def receive(message: Any): AnyRef = message match {
    case WorkerSample(execId, cpu, heap) => ProcessCpuPlugin.report(execId, cpu, heap); null
    case _                               => null
  }
}

final class ProcessCpuExecutorPlugin extends ExecutorPlugin {
  private var ctx: PluginContext                  = _
  private var osBean: OperatingSystemMXBean       = _
  private var startCpu: Long                      = 0L
  private var scheduler: ScheduledExecutorService = _

  override def init(c: PluginContext, extraConf: java.util.Map[String, String]): Unit = {
    ctx      = c
    osBean   = ManagementFactory.getPlatformMXBean(classOf[OperatingSystemMXBean])
    startCpu = osBean.getProcessCpuTime
    // Reset the JVM's heap-pool peak counters so getPeakUsage reflects THIS run
    // only — not memory used before the plugin started (e.g. a prior run in a
    // reused local-mode JVM).
    heapPools.foreach(_.resetPeakUsage())
    scheduler = Executors.newSingleThreadScheduledExecutor { r =>
      val t = new Thread(r, "worker-resource-reporter"); t.setDaemon(true); t
    }
    // Periodic sends keep the driver current even if shutdown is abrupt (killed
    // executor): CPU delta is monotonic and heap is a JVM-tracked peak, so the
    // driver's per-executor max holds regardless of when the last send landed.
    scheduler.scheduleAtFixedRate(() => sample(), 1L, 1L, TimeUnit.SECONDS)
  }

  /** Heap memory pools only (Eden/Survivor/Old); non-heap (Metaspace, code cache)
   *  is excluded so the figure matches Flink's Heap.Used gauge. */
  private def heapPools =
    ManagementFactory.getMemoryPoolMXBeans.asScala.filter(_.getType == MemoryType.HEAP)

  /** Peak heap used since the last reset, summed across heap pools [bytes]. */
  private def peakHeapUsed(): Long =
    heapPools.foldLeft(0L) { (acc, pool) =>
      acc + Option(pool.getPeakUsage).map(_.getUsed).getOrElse(0L)
    }

  private def sample(): Unit = {
    if (ctx == null || osBean == null) return
    val cpu      = osBean.getProcessCpuTime            // -1 if the platform can't report it
    val cpuDelta = if (cpu > 0L) math.max(0L, cpu - startCpu) else 0L
    // Send unconditionally so heap is recorded even when CPU is unavailable.
    try ctx.send(WorkerSample(ctx.executorID(), cpuDelta, peakHeapUsed()))
    catch { case NonFatal(_) => () }  // best-effort
  }

  override def shutdown(): Unit = {
    try sample()
    finally if (scheduler != null) scheduler.shutdownNow()
  }
}
