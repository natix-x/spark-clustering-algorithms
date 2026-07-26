package clustering.algorithms.kmeans

import clustering.core.NearestPrototypeModel
import clustering.distance.DistanceMetric
import org.apache.spark.ml.linalg.Vector


/** A fitted k-means model: each point is labelled with its nearest centroid. */
class KMeansModel(
  centroidsArg: Array[Vector],
  distanceArg:  DistanceMetric
) extends NearestPrototypeModel(centroidsArg, distanceArg) {

  /** The cluster centroids, in cluster-id order. */
  def centroids: Array[Vector] = prototypes
}
