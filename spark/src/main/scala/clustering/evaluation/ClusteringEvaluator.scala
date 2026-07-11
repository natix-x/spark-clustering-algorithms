package clustering.evaluation

import clustering.core.Model
import org.apache.spark.sql.DataFrame


trait ClusteringEvaluator extends Serializable {
  def evaluate(model: Model, data: DataFrame): Double
}
