package clustering.algorithms.kmeans

import clustering.core.Model
import clustering.data.Point
import clustering.distance.DistanceMetric
import clustering.utils.MathUtils


class KMeansModel(
  val centroids: Array[Point],
  val distance: DistanceMetric
) extends Model {

  def predict(point: Point): Int =
    MathUtils.argmin(centroids)(distance.compute(_, point))
}
