package clustering.algorithms.dbscan

import clustering.distance.DistanceMetric
import clustering.utils.PartitionAggregator
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.rdd.RDD
import org.log4s.getLogger

/** For every candidate point, the weighted number of dataset points within ε of it.
 *
 *  Shared by `dbscanpp` (core-point test `density ≥ minPts`) and `dpc` (local density ρ).
 *  Counted against ALL n points, so densities are exact; only the candidate set is sampled.
 *
 *  Candidates go out in broadcast chunks of `chunkSize`, one Spark job (= one full pass) per
 *  chunk, folded per partition via [[PartitionAggregator]]. Bigger chunks = fewer passes, but
 *  the broadcast scales with chunk × dimensionality (2 000 × 1024 dims ≈ 16 MB); raise it on
 *  low-dimensional data. Caller should cache its input.
 *
 *  Cost: n × m distance computations.
 */
object EpsilonNeighbourCounter {

  private val logger = getLogger

  /** `result(i)` = Σ { w_x : x ∈ points, d(candidates(i), x) ≤ eps }.
   *
   *  A candidate that is itself a dataset point counts itself (DBSCAN core condition).
   *  Counts are weighted, so `minPts` is a threshold on mass.
   */
  def computeNeighbourhoodDensities(
    datasetPoints: RDD[(Vector, Double)],
    candidatePoints: Array[Vector],
    eps: Double,
    distanceMetric: DistanceMetric,
    chunkSize: Int
  ): Array[Double] = {
    val sc = datasetPoints.sparkContext
    val aggregatedCounts = new Array[Double](candidatePoints.length)
    var currentOffset = 0

    // One line per chunk, i.e. per full pass over the data: chunk count is m/chunkSize, so
    // this is one line for a small run and a steady heartbeat for a long one. Spark's own
    // logs only show the broadcast, which says nothing about how many passes remain.
    val chunkCount = (candidatePoints.length + chunkSize - 1) / chunkSize
    val startedAt  = System.nanoTime()
    var chunkIndex = 0

    candidatePoints.grouped(chunkSize).foreach { candidateChunk =>
      val bcChunk = sc.broadcast(candidateChunk)
      val distance = distanceMetric
      val radius = eps
      val partialCounts = PartitionAggregator.aggregateDoubles(datasetPoints, candidateChunk.length) { (accumulator, row) =>
        val (pointCoords, pointWeight) = row
        val currentChunk = bcChunk.value
        var candidateIndex = 0
        while (candidateIndex < currentChunk.length) {
          if (distance.compute(currentChunk(candidateIndex), pointCoords) <= radius) accumulator(candidateIndex) += pointWeight
          candidateIndex += 1
        }
      }
      bcChunk.destroy()
      System.arraycopy(partialCounts, 0, aggregatedCounts, currentOffset, candidateChunk.length)
      currentOffset += candidateChunk.length
      chunkIndex += 1

      val elapsed = (System.nanoTime() - startedAt) / 1e9
      logger.info(f"epsilon-counting: chunk $chunkIndex%d/$chunkCount%d (${candidateChunk.length}%d candidates), " +
        f"$elapsed%.0f s elapsed, ${elapsed * (chunkCount - chunkIndex) / chunkIndex}%.0f s left")
    }

    aggregatedCounts
  }
}
