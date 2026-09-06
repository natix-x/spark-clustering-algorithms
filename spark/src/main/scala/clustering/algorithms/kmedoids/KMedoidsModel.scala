package clustering.algorithms.kmedoids

import clustering.core.NearestPrototypeModel
import clustering.distance.DistanceMetric
import org.apache.spark.ml.linalg.Vector


class KMedoidsModel(
  medoidsArg:  Array[Vector],
  distanceArg: DistanceMetric
) extends NearestPrototypeModel(medoidsArg, distanceArg) {

  /** The cluster medoids, in cluster-id order. */
  def medoids: Array[Vector] = prototypes
}