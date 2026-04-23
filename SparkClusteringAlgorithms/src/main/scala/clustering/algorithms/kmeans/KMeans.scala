package clustering.algorithms.kmeans

import clustering.data.{DatasetOps, Point}
import clustering.distance.DistanceMetric
import clustering.utils.{Convergence, MathUtils, SparkUtils}
import clustering.core.Clusterer
import clustering.distance.EuclideanDistance
import org.apache.spark.rdd.RDD


class KMeans(
  val k: Int,
  val maxIter: Int = 100,
  val eps: Double = 1e-4,
  val distance: DistanceMetric = new EuclideanDistance()
) extends Clusterer {

  override def fit(data: RDD[Point]): KMeansModel = {
    val cached = DatasetOps.cachePoints(data)
    implicit val sc = cached.sparkContext

    var centroids = cached.takeSample(withReplacement = false, num = k, seed = 42L)
    var iter      = 0
    var converged = false

    while (!converged && iter < maxIter) {
      val bcCentroids = SparkUtils.broadcastSafe(centroids)

      val newCentroidMap: Map[Int, Point] = cached
        .map { p =>
          val ci = MathUtils.argmin(bcCentroids.value)(distance.compute(_, p))
          (ci, p)
        }
        .groupByKey()
        .mapValues(DatasetOps.computeCentroid)
        .collectAsMap()
        .toMap

      bcCentroids.destroy()

      val newCentroids = (0 until k).map { i =>
        newCentroidMap.getOrElse(i, centroids(i))
      }.toArray

      converged  = Convergence.hasConverged(centroids, newCentroids, eps, distance)
      centroids  = newCentroids
      iter      += 1
    }

    new KMeansModel(centroids, distance)
  }
}
