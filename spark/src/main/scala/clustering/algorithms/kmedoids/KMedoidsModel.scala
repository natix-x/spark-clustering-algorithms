package clustering.algorithms.kmedoids

import clustering.core.NearestPrototypeModel
import clustering.distance.DistanceMetric
import org.apache.spark.ml.linalg.Vector


/** A fitted k-medoids model: each point is labelled with its nearest medoid. */
class KMedoidsModel(
  medoidsArg:  Array[Vector],
  distanceArg: DistanceMetric
) extends NearestPrototypeModel(medoidsArg, distanceArg) {

  /** The cluster medoids, in cluster-id order. */
  def medoids: Array[Vector] = prototypes
}