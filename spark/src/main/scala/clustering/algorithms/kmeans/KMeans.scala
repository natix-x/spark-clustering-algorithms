package clustering.algorithms.kmeans

import clustering.core.{Clusterer, EuclideanGeometry, Geometry}
import org.apache.spark.sql.DataFrame
import org.apache.spark.storage.StorageLevel

// Lloyd's k-means or spherical k-means (Dhillon & Modha 2001) based on the geometry.

class KMeans(
    val k: Int,
    val maxIter: Int = 100,
    val eps: Double = 1e-4,
    val seed: Long = 42L,
    val geometry: Geometry = EuclideanGeometry
) extends Clusterer {

  override def fit(data: DataFrame): KMeansModel = {
    val setup = LloydKMeans.initialize(data, geometry, k, seed, StorageLevel.MEMORY_AND_DISK)
    val centroids = LloydKMeans.run(setup.preparedPoints, setup.initialCentroids, setup.fitDistance, geometry, maxIter, eps)
    setup.preparedPoints.unpersist(blocking = false)
    new KMeansModel(centroids, geometry.modelDistance)
  }
}
