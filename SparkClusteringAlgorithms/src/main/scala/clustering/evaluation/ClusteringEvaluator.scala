package clustering.evaluation

import clustering.core.Model
import clustering.data.Point
import org.apache.spark.rdd.RDD


trait ClusteringEvaluator extends Serializable {
  def evaluate(model: Model, data: RDD[Point]): Double
}
