package clustering.algorithms.dbscan

import clustering.distance.DistanceMetric
import clustering.utils.PartitionAggregator
import org.apache.spark.SparkConf
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.rdd.RDD
import org.log4s.getLogger


/** For every candidate point, the weighted number of dataset points within ε of it — step 2 of
 *  DBSCAN++.
 *
 *  Cost: n × m distance computations.
 */
object EpsilonNeighbourCounter {

  private val logger = getLogger

  private val HARD_POINT_LIMIT = 50000 // do not allow more points

  /** As many candidates as fit the broadcast budget (capped at `HARD_POINT_LIMIT`), split evenly
   *  so no chunk is left with only a handful of points (each chunk costs a full scan of
   *  `datasetPoints` regardless of how many candidates it carries). */
  private[dbscan] def resolveChunkSize(dimension: Int, candidateCount: Int, conf: SparkConf): Int = {
    val bytesPerCandidate = dimension.toLong * 8L
    val maxBroadcastBytes = EpsilonGraphComponents.getMaxBroadcastBytes(conf)
    val capacity = math.min(HARD_POINT_LIMIT, math.max(1, (maxBroadcastBytes / bytesPerCandidate).toInt))
    val chunkCount = math.max(1, math.ceil(candidateCount.toDouble / capacity).toInt)
    math.ceil(candidateCount.toDouble / chunkCount).toInt
  }

  /** `result(i)` = Σ { w_x : x ∈ points, d(candidates(i), x) ≤ eps }. Weighted, so `minPts` is a
   *  threshold on mass, and a candidate that is itself a dataset point counts itself.
   *
   *  Chunk size is picked automatically — see [[resolveChunkSize]]. */
  def computeNeighbourhoodDensities(
    datasetPoints: RDD[(Vector, Double)],
    candidatePoints: Array[Vector],
    eps: Double,
    distanceMetric: DistanceMetric
  ): Array[Double] = {
    if (candidatePoints.isEmpty) return Array.empty[Double]

    val sc = datasetPoints.sparkContext
    val resolvedChunkSize = resolveChunkSize(candidatePoints(0).size, candidatePoints.length, sc.getConf)
    logger.info(s"dbscanpp: step 2 chunkSize=$resolvedChunkSize (m=${candidatePoints.length})")
    val totalDensities = new Array[Double](candidatePoints.length)
    val chunkCount = (candidatePoints.length + resolvedChunkSize - 1) / resolvedChunkSize
    val startedAt = System.nanoTime()

    var currentOffset = 0
    var chunkIndex = 0

    candidatePoints.grouped(resolvedChunkSize).foreach { candidateChunk =>
      val rawCandidates = candidateChunk.map(_.toArray)
      val broadcastChunk = sc.broadcast(rawCandidates)

      val distance = distanceMetric
      val radius = eps

      val partialDensities = PartitionAggregator.aggregateDoubles(datasetPoints, candidateChunk.length) { (accumulator, row) =>
        val (pointCoords, pointWeight) = row
        val pointArray = pointCoords.toArray
        val currentChunk = broadcastChunk.value

        var candidateIndex = 0
        while (candidateIndex < currentChunk.length) {
          if (distance.withinRadius(currentChunk(candidateIndex), pointArray, radius)) {
            accumulator(candidateIndex) += pointWeight
          }
          candidateIndex += 1
        }
      }
      broadcastChunk.destroy()
      System.arraycopy(partialDensities, 0, totalDensities, currentOffset, candidateChunk.length)
      currentOffset += candidateChunk.length
      chunkIndex += 1

      logProgress(chunkIndex, chunkCount, candidateChunk.length, startedAt)
    }

    totalDensities
  }

  private def logProgress(chunkIndex: Int, chunkCount: Int, currentChunkSize: Int, startedAt: Long): Unit = {
    val elapsed = (System.nanoTime() - startedAt) / 1e9
    val estimatedSecondsLeft = if (chunkIndex > 0) elapsed * (chunkCount - chunkIndex) / chunkIndex else 0.0

    logger.info(f"epsilon-counting: chunk $chunkIndex%d/$chunkCount%d ($currentChunkSize%d candidates), " +
      f"$elapsed%.0f s elapsed, $estimatedSecondsLeft%.0f s left")
  }
}
