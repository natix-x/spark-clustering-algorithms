package clustering.core

import org.apache.spark.rdd.RDD
import clustering.data.Point


trait Clusterer extends Serializable {
  def fit(data: RDD[Point]): Model
}
