package clustering.evaluation

import clustering.core.Model
import clustering.distance.{DistanceMetric, EuclideanDistance}
import org.apache.spark.sql.DataFrame

/** Davies-Bouldin index (Davies & Bouldin 1979).
 *
 *  {{{
 *  DB = (1/k) Σ_i max_{j≠i} (S_i + S_j) / M_ij
 *  }}}
 *
 *  `S_i` = mean distance of cluster i's points to its centroid, `M_ij` = distance between
 *  centroids i and j: each cluster scored against its worst (most similar) neighbour. **Lower
 *  is better**, 0 is the floor — opposite direction to the silhouette and [[CalinskiHarabaszIndex]].
 *
 *  O(n·d + k²·d) vs the silhouette's O(n²·d), so it runs on the FULL dataset — but it only sees
 *  points-to-centroids, not cluster shape.
 */
class DaviesBouldinEvaluator(val distance: DistanceMetric = EuclideanDistance) extends ClusteringEvaluator {

  override def evaluate(model: Model, data: DataFrame): Double =
    DaviesBouldinIndex.of(ClusterMoments.compute(model, data, distance), distance)
}

object DaviesBouldinIndex {

  /** The index, from moments already computed. Driver-side, O(k²·d).
   *
   *  Degenerate cases follow scikit-learn's `davies_bouldin_score`:
   *   - fewer than two clusters — undefined, reported as 0.0;
   *   - coincident centroids (`M_ij = 0`) — pair skipped rather than contributing `Infinity`
   *     (not valid JSON); a cluster with every neighbour skipped scores 0.
   */
  def of(moments: ClusterMoments, distance: DistanceMetric): Double = {
    val clusterMoments = moments.clusterMoments
    val numClusters = clusterMoments.length
    if (numClusters < 2) return 0.0

    val clusterDispersions = clusterMoments.map(_.meanDistance)
    val clusterCentroids = clusterMoments.map(_.centroid.toArray)

    // Centroid distance matrix, upper triangle only — d is symmetric and the diagonal is unused.
    val centroidDistances = Array.ofDim[Double](numClusters, numClusters)
    var i = 0
    while (i < numClusters) {
      var j = i + 1
      while (j < numClusters) {
        val distanceBetweenCentroids = distance.compute(clusterCentroids(i), clusterCentroids(j))
        centroidDistances(i)(j) = distanceBetweenCentroids
        centroidDistances(j)(i) = distanceBetweenCentroids
        j += 1
      }
      i += 1
    }

    var totalWorstCaseOverlap = 0.0
    i = 0
    while (i < numClusters) {
      var maxSimilarityRatio = 0.0
      var j = 0
      while (j < numClusters) {
        if (j != i) {
          val distanceBetweenCentroids = centroidDistances(i)(j)
          if (distanceBetweenCentroids > 0.0) {
            val similarityRatio = (clusterDispersions(i) + clusterDispersions(j)) / distanceBetweenCentroids
            if (similarityRatio > maxSimilarityRatio) maxSimilarityRatio = similarityRatio
          }
        }
        j += 1
      }
      totalWorstCaseOverlap += maxSimilarityRatio
      i += 1
    }

    totalWorstCaseOverlap / numClusters
  }
}
