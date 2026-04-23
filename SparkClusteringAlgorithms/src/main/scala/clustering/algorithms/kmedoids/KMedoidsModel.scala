package clustering.algorithms.kmedoids

import clustering.core.Model
import clustering.data.Point
import clustering.distance.DistanceMetric
import clustering.utils.MathUtils


class KMedoidsModel(
  val medoids: Array[Point],
  val distance: DistanceMetric
) extends Model {

  def predict(point: Point): Int =
    MathUtils.argmin(medoids)(distance.compute(_, point))
}
