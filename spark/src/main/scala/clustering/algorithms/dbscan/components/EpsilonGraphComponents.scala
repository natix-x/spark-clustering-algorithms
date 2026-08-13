package clustering.algorithms.dbscan.components

import clustering.algorithms.dbscan.utils.{ExecutionPlan, ScanProgress}
import clustering.distance.DistanceMetric
import clustering.utils.UnionFind
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.network.util.JavaUtils
import org.apache.spark.{SparkConf, SparkContext}
import org.log4s.getLogger

import scala.collection.mutable

/** Connected components of the ε-graph over the core points — step 3 of DBSCAN++.
 */
private[dbscan] object EpsilonGraphComponents {

  private val logger = getLogger

  // configs
  private val ROWS_PER_BLOCK = 128
  private val MAX_ROWS_PER_DISTRIBUTED_BLOCK = 4096
  private val MIN_CORE_POINTS_TO_DISTRIBUTE = 20000
  private val BROADCAST_MEMORY_FRACTION = 0.3
  private val MAX_BLOCKS = 2000L

  private[dbscan] def getMaxBroadcastBytes(conf: SparkConf): Long = {
    val executorMemoryBytes = JavaUtils.byteStringAsBytes(conf.get("spark.executor.memory", "1g"))
    (executorMemoryBytes * BROADCAST_MEMORY_FRACTION).toLong
  }

  /** Cluster id per core point, contiguous from 0, numbered by ascending minimum member index. */
  def compute(cores: Array[Vector], eps: Double, metric: DistanceMetric, sc: SparkContext): Array[Int] = {
    val broadcastBytes = cores.length.toLong * dimensionOf(cores) * 8L
    val maxBroadcastBytes = getMaxBroadcastBytes(sc.getConf)
    val executionPlan = planFor(cores.length, broadcastBytes, maxBroadcastBytes)

    executionPlan match {
      case ExecutionPlan.Distributed(rowsPerBlock) =>
        logger.info(f"dbscanpp: step 3 ε-graph distributed — ${cores.length}%d cores, " +
          f"$rowsPerBlock%d rows/block, broadcast ${broadcastBytes / 1e6}%.0f MB")
      case ExecutionPlan.DriverLocal(reason) =>
        logger.info(s"dbscanpp: step 3 ε-graph on the driver — $reason")
    }

    val startedAt = System.nanoTime()
    val labels = executionPlan match {
      case ExecutionPlan.Distributed(rowsPerBlock) => computeDistributed(cores, eps, metric, sc, rowsPerBlock)
      case ExecutionPlan.DriverLocal(_) => computeLocal(cores, eps, metric)
    }
    val elapsed = (System.nanoTime() - startedAt) / 1e9
    logger.info(f"dbscanpp: step 3 ε-graph done in $elapsed%.0f s over ${cores.length}%d cores")
    labels
  }

  /** Chooses the path. Pure, so unit-testable without a cluster — see `docs/dbscanpp_docs.md` §4
   *  for why graph density is not one of the criteria any more. */
  private[dbscan] def planFor(
    coreCount: Int,
    broadcastBytes: Long,
    maxBroadcastBytes: Long
  ): ExecutionPlan = {
    if (coreCount < MIN_CORE_POINTS_TO_DISTRIBUTE) {
      ExecutionPlan.DriverLocal(s"only $coreCount cores (below $MIN_CORE_POINTS_TO_DISTRIBUTE), the scan is cheaper than the fixed costs")
    } else if (broadcastBytes > maxBroadcastBytes) {
      ExecutionPlan.DriverLocal(f"broadcasting $coreCount%d cores would take ${broadcastBytes / 1e9}%.1f GB per executor " +
        f"(cap ${maxBroadcastBytes / 1e9}%.1f GB) — lower coreSampleFraction if this phase dominates")
    } else {
      ExecutionPlan.Distributed(math.max(MAX_ROWS_PER_DISTRIBUTED_BLOCK, coreCount / MAX_BLOCKS).toInt)
    }
  }

  /** Parallel over the driver's cores; unions run after each block, sequentially. */
  private[dbscan] def computeLocal(cores: Array[Vector], eps: Double, metric: DistanceMetric): Array[Int] = {
    val unionFind = new UnionFind(cores.length)
    val progress = new ScanProgress(cores.length)
    val rawCoords = extractRawCoordinates(cores)
    var blockStart = 0

    while (blockStart < cores.length) {
      val blockEnd = math.min(blockStart + ROWS_PER_BLOCK, cores.length)
      val neighboursPerRow = (blockStart until blockEnd).par
        .map(row => findNeighborsAboveIndex(row, rawCoords, eps, metric))
        .toArray

      var row = blockStart
      while (row < blockEnd) {
        applyEdges(unionFind, row, neighboursPerRow(row - blockStart))
        row += 1
      }
      blockStart = blockEnd
      progress.report(blockStart, "driver-local")
    }

    unionFind.componentIds()
  }

  /** Broadcast the cores, scan an index range on the cluster, union the collected edges on the
   *  driver. Each partition compacts its own edges through a local `UnionFind` before shipping
   *  them — see `docs/dbscanpp_docs.md` §4. */
  private[dbscan] def computeDistributed(
    cores: Array[Vector],
    eps: Double,
    distanceMetric: DistanceMetric,
    sc: SparkContext,
    rowsPerBlock: Int = MAX_ROWS_PER_DISTRIBUTED_BLOCK
  ): Array[Int] = {
    val unionFind = new UnionFind(cores.length)
    val progress = new ScanProgress(cores.length)
    val broadcastCores = sc.broadcast(extractRawCoordinates(cores))
    val coreCount = cores.length

    try {
      var blockStart = 0
      while (blockStart < cores.length) {
        val blockEnd = math.min(blockStart + rowsPerBlock, cores.length)
        // Row i scans m − i others, so a contiguous block is uneven work; more slices than cores
        // lets the scheduler even it out.
        val slices = math.max(1, math.min(blockEnd - blockStart, sc.defaultParallelism * 4))

        val mergesPerPartition = sc
          .parallelize(blockStart until blockEnd, slices)
          .mapPartitions { rows =>
            val partitionUnionFind = new UnionFind(coreCount)
            val merges = mutable.ArrayBuilder.make[Int]

            rows.foreach { row =>
              val neighbours = findNeighborsAboveIndex(row, broadcastCores.value, eps, distanceMetric)
              var t = 0
              while (t < neighbours.length) {
                if (partitionUnionFind.union(row, neighbours(t))) {
                  merges += row
                  merges += neighbours(t)
                }
                t += 1
              }
            }
            Iterator(merges.result())
          }
          .collect()

        // Order is irrelevant: `UnionFind.union` attaches the larger root under the smaller, so the
        // components — and the min-index labels drawn from them — depend only on the edge SET.
        mergesPerPartition.foreach { merges =>
          var i = 0
          while (i < merges.length) {
            unionFind.union(merges(i), merges(i + 1))
            i += 2
          }
        }
        blockStart = blockEnd
        progress.report(blockStart, "distributed")
      }
    } finally {
      broadcastCores.destroy()
    }

    unionFind.componentIds()
  }

  /** Indices `j > row` within ε of `row`, ascending. Raw coordinate arrays — see
   *  `docs/dbscanpp_docs.md` §3 for why this beats `Vector`-typed comparisons. */
  private def findNeighborsAboveIndex(
    row: Int,
    corePoints: Array[Array[Double]],
    eps: Double,
    distanceMetric: DistanceMetric
  ): Array[Int] = {
    val neighbours = mutable.ArrayBuilder.make[Int]
    val rowCoords  = corePoints(row)
    var j = row + 1

    while (j < corePoints.length) {
      if (distanceMetric.withinRadius(rowCoords, corePoints(j), eps)) {
        neighbours += j
      }
      j += 1
    }
    neighbours.result()
  }

  private def extractRawCoordinates(cores: Array[Vector]): Array[Array[Double]] =
    cores.map(_.toArray)

  private def applyEdges(unionFind: UnionFind, row: Int, neighbours: Array[Int]): Unit = {
    var t = 0
    while (t < neighbours.length) {
      unionFind.union(row, neighbours(t))
      t += 1
    }
  }

  private def dimensionOf(cores: Array[Vector]): Int =
    if (cores.isEmpty) 0 else cores(0).size

}
