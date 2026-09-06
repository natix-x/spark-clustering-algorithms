package clustering.evaluation

import clustering.core.Model
import clustering.distance.{DistanceMetric, EuclideanDistance}
import org.apache.spark.sql.DataFrame

/** Calinski-Harabasz index, a.k.a. the variance ratio criterion (Calinski & Harabasz 1974).
 *
 *  {{{
 *  CH = [ Σ_i n_i·d(c_i, c)² / (k-1) ] / [ Σ_i Σ_{x∈C_i} d(x, c_i)² / (n-k) ]
 *  }}}
 *
 *  Between-cluster dispersion over within-cluster dispersion, each over its degrees of freedom.
 *  **Higher is better**, unbounded above — only comparable across labellings of the SAME data,
 *  not across datasets.
 *
 *  Same formula on whatever metric's squared distances, so a k-medoids run under `manhattan` is
 *  scored under `manhattan` — consistent with the algorithm's own objective, at the cost of the
 *  Euclidean ANOVA reading.
 *
 *  O(n·d) like [[DaviesBouldinIndex]]: computed on the FULL dataset, never a sample.
 */
class CalinskiHarabaszEvaluator(val distance: DistanceMetric = EuclideanDistance) extends ClusteringEvaluator {

  override def evaluate(model: Model, data: DataFrame): Double =
    CalinskiHarabaszIndex.of(ClusterMoments.compute(model, data, distance), distance)
}

object CalinskiHarabaszIndex {

  /** The index, from moments already computed — driver-side arithmetic over k clusters.
   *
   *  `totalDataWeight` is total WEIGHT of non-noise points, not row count (== row count on
   *  unweighted input).
   *
   *  Degenerate cases follow scikit-learn's `calinski_harabasz_score`:
   *   - fewer than two clusters, or `totalDataWeight <= numClusters` — undefined, reported 0.0;
   *   - zero within-cluster dispersion — 1.0 (ratio's denominator vanishes).
   */
  def of(moments: ClusterMoments, distance: DistanceMetric): Double = {
    val clusterMoments = moments.clusterMoments
    val numClusters = clusterMoments.length
    val totalDataWeight = moments.totalWeight

    if (numClusters < 2 || totalDataWeight <= numClusters) return 0.0

    val withinClusterDispersion = clusterMoments.foldLeft(0.0)(_ + _.sumOfWeightedSquaredDistances)

    // `!isFinite` catches NaN as well as infinity: a NaN coordinate propagates through the
    // moments, and NaN fails every `<=` test, so it used to reach the emitted result — where it
    // is serialised as a bare `NaN` token that is not valid JSON, losing the whole run's file.
    // `java.lang.Double.isFinite`, not `x.isFinite`: the latter is Scala 2.13's RichDouble.
    if (!java.lang.Double.isFinite(withinClusterDispersion) || withinClusterDispersion <= 0.0) return 1.0

    val globalCentroid = moments.globalCentroid.toArray

    val betweenClusterDispersion = clusterMoments.foldLeft(0.0) { (accumulatedDispersion, cluster) =>
      val distanceToGlobalCentroid = distance.compute(cluster.centroid.toArray, globalCentroid)
      accumulatedDispersion + (cluster.clusterWeight * distanceToGlobalCentroid * distanceToGlobalCentroid)
    }

    val betweenClusterVariance = betweenClusterDispersion / (numClusters - 1)
    val withinClusterVariance = withinClusterDispersion / (totalDataWeight - numClusters)

    val index = betweenClusterVariance / withinClusterVariance
    if (java.lang.Double.isFinite(index)) index else 0.0
  }
}