package clustering.core

import org.apache.spark.sql.DataFrame


trait Clusterer extends Serializable {
  def fit(data: DataFrame): Model
}
