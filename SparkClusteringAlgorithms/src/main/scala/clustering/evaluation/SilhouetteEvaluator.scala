package clustering.evaluation

import clustering.core.Model
import clustering.data.Point
import clustering.distance.{DistanceMetric, EuclideanDistance}
import clustering.utils.SparkUtils
import org.apache.spark.rdd.RDD


/** Computes the mean silhouette score over all non-noise points.
 *
 *  s(p) = (b - a) / max(a, b)
 *  where a = mean intra-cluster distance, b = mean nearest-cluster distance.
 *  Score in [-1, 1]; higher is better.
 *
 *  Noise points (label -1, as produced by DBSCAN) are excluded.
 */
class SilhouetteEvaluator(
                           val distance: DistanceMetric = new EuclideanDistance()
                         ) extends ClusteringEvaluator {

  override def evaluate(model: Model, data: RDD[Point]): Double = {
    implicit val sc = data.sparkContext

    val labeled: RDD[(Point, Int)] = data.map(p => (p, model.predict(p)))

    val clusterMap: Map[Int, Array[Point]] = labeled
      .filter { case (_, l) => l != -1 }
      .map    { case (p, l) => (l, p)  }
      .groupByKey()
      .collectAsMap()
      .map { case (l, pts) => l -> pts.toArray }
      .toMap

    val bcClusters = SparkUtils.broadcastSafe(clusterMap)

    val (scoreSum, count) = labeled
      .filter { case (_, l) => l != -1 }
      .map { case (p, label) =>
        val clus        = bcClusters.value
        val sameCluster = clus.getOrElse(label, Array.empty).filter(_ != p)
        val a           = meanDist(p, sameCluster)
        val b           = clus
          .filter { case (k, _) => k != label }
          .values
          .map(pts => meanDist(p, pts))
          .reduceOption(_ min _)
          .getOrElse(Double.MaxValue)
        val denom = math.max(a, b)
        if (denom == 0.0) 0.0 else (b - a) / denom
      }
      .aggregate((0.0, 0L))(
        { case ((s, c), v) => (s + v, c + 1L) },
        { case ((s1, c1), (s2, c2)) => (s1 + s2, c1 + c2) }
      )

    bcClusters.destroy()
    if (count == 0L) 0.0 else scoreSum / count
  }

  private def meanDist(p: Point, pts: Array[Point]): Double =
    if (pts.isEmpty) 0.0
    else pts.map(q => distance.compute(p, q)).sum / pts.length
}
