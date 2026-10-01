package clustering.algorithms.kmeans

import clustering.core.NearestPrototypeModel
import clustering.distance.DistanceMetric
import org.apache.spark.ml.linalg.Vector


class KMeansModel(
  centroidsArg: Array[Vector],
  distanceArg: DistanceMetric,
  iterationsRunArg: Option[Int] = None
) extends NearestPrototypeModel(centroidsArg, distanceArg) {

  /** The cluster centroids, in cluster-id order. */
  def centroids: Array[Vector] = prototypes

  override val iterationsRun: Option[Int] = iterationsRunArg
}
