package clustering.utils

import clustering.distance.DistanceMetric
import org.apache.spark.ml.linalg.Vector


object Convergence {

  def hasConverged(
                    oldCentroids: Array[Vector],
                    newCentroids: Array[Vector],
                    eps: Double,
                    distance: DistanceMetric
                  ): Boolean = {

    oldCentroids.zip(newCentroids).forall { case (o, n) =>
      distance.compute(o, n) < eps
    }
  }
}