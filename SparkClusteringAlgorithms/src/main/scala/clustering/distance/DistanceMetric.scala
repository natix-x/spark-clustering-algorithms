package clustering.distance

import org.apache.spark.ml.linalg.Vector


trait DistanceMetric extends Serializable {
  def compute(a: Vector, b: Vector): Double
}