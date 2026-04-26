package clustering.algorithms.dbscan

import clustering.core.Model
import clustering.data.Point
import clustering.distance.DistanceMetric
import org.apache.spark.rdd.RDD


class DBSCANModel(
  val labelsRDD: RDD[(Point, Int)],
  val corePoints: Set[Point],
  val eps: Double,
  val distance: DistanceMetric
) extends Model {

  @transient private lazy val labelsMap: Map[Point, Int] =
    labelsRDD.collectAsMap().toMap

  override def labeledData(data: RDD[Point]): RDD[(Point, Int)] = labelsRDD

  def predict(point: Point): Int =
    labelsMap.getOrElse(point, predictUnseen(point))

  private def predictUnseen(point: Point): Int =
    corePoints
      .find(cp => distance.compute(cp, point) <= eps)
      .flatMap(labelsMap.get)
      .getOrElse(-1)
}
