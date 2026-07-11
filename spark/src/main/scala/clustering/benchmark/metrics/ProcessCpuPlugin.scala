package clustering.benchmark.metrics

import com.sun.management.OperatingSystemMXBean
import org.apache.spark.api.plugin.{DriverPlugin, ExecutorPlugin, PluginContext, SparkPlugin}

import java.lang.management.ManagementFactory
import java.util.concurrent.{ConcurrentHashMap, Executors, ScheduledExecutorService, TimeUnit}
import scala.jdk.CollectionConverters._

/** RPC message: one executor's process-CPU delta [ns] (since plugin init) and its
 *  current JVM heap-used [bytes]. Top-level (not nested) so it serializes cleanly
 *  over the plugin RPC channel. */
private[metrics] final case class WorkerSample(execId: String, cpuNanos: Long, heapBytes: Long)

/** Cross-framework-comparable engine resource metrics. Spark's standard listener
 *  gives PER-TASK CPU (`TaskMetrics.executorCpuTime`) and delivers no executor
 *  memory updates in local mode — neither is comparable with Flink, where the only
 *  CPU/RAM signals are WHOLE-JVM gauges (`Status.JVM.CPU.Time`,
 *  `Status.JVM.Memory.Heap.Used`). This plugin samples the SAME JVM MXBeans on
 *  every worker JVM, mirroring Flink's reporter, without forking Spark.
 *
 *  Each executor periodically samples process CPU (`getProcessCpuTime`) and heap
 *  used (`MemoryMXBean`) and reports to the driver; the driver keeps the max per
 *  executor (CPU delta and heap are both taken as the peak). Aggregates SUM across
 *  executors — matching the Flink side, which sums per-TaskManager values. In
 *  `local[*]` the single JVM is driver+executor (includes driver), exactly as
 *  Flink's local MiniCluster. Register via
 *  {@code spark.plugins=clustering.benchmark.metrics.ProcessCpuPlugin}. */
object ProcessCpuPlugin {
  private val cpuByExecutor  = new ConcurrentHashMap[String, java.lang.Long]()
  private val heapByExecutor = new ConcurrentHashMap[String, java.lang.Long]()

  private[metrics] def report(execId: String, cpuNanos: Long, heapBytes: Long): Unit = {
    cpuByExecutor.merge(execId, cpuNanos, (a, b) => Math.max(a, b))
    heapByExecutor.merge(execId, heapBytes, (a, b) => Math.max(a, b))
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
    scheduler = Executors.newSingleThreadScheduledExecutor { r =>
      val t = new Thread(r, "worker-resource-reporter"); t.setDaemon(true); t
    }
    // Periodic sends keep the driver current even if shutdown is abrupt; CPU delta
    // is monotonic and heap is taken as peak, so the driver's per-executor max holds.
    scheduler.scheduleAtFixedRate(() => sample(), 1L, 1L, TimeUnit.SECONDS)
  }

  private def sample(): Unit = {
    if (ctx == null || osBean == null) return
    val cpu  = osBean.getProcessCpuTime
    val heap = ManagementFactory.getMemoryMXBean.getHeapMemoryUsage.getUsed
    if (cpu > 0L) {
      try ctx.send(WorkerSample(ctx.executorID(), cpu - startCpu, heap))
      catch { case _: Throwable => () }  // best-effort
    }
  }

  override def shutdown(): Unit = {
    try sample()
    finally if (scheduler != null) scheduler.shutdownNow()
  }
}
