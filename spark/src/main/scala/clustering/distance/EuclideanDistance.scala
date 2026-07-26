package clustering.distance

import org.apache.spark.ml.linalg.{Vector, Vectors}

object EuclideanDistance extends DistanceMetric {

  override def compute(a: Vector, b: Vector): Double = {
    math.sqrt(Vectors.sqdist(a, b))
  }
}
