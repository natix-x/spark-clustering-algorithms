package clustering.algorithms.dbscan

import clustering.core.Model
import clustering.data.Point
import clustering.distance.DistanceMetric


/** DBSCAN model produced by both DBSCAN and GridDBSCAN.
 *
 *  Noise points are labelled -1.
 *  For unseen points: assigns the cluster of the nearest core point
 *  within eps, or -1 if none exists.
 */
class DBSCANModel(
  val labels: Map[Point, Int],
  val corePoints: Set[Point],
  val eps: Double,
  val distance: DistanceMetric
) extends Model {

  def predict(point: Point): Int =
    labels.getOrElse(point, predictUnseen(point))

  private def predictUnseen(point: Point): Int =
    corePoints
      .find(cp => distance.compute(cp, point) <= eps)
      .flatMap(labels.get)
      .getOrElse(-1)
}
