package clustering.utils

import clustering.data.Point
import clustering.distance.DistanceMetric


object Convergence {

  def hasConverged(
      oldCentroids: Array[Point],
      newCentroids: Array[Point],
      eps: Double,
      distance: DistanceMetric
    ): Boolean = {

    oldCentroids.zip(newCentroids).forall { case (o, n) =>
      distance.compute(o, n) < eps
    }
  }
}
