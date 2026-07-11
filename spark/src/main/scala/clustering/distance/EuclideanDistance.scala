package clustering.distance

import org.apache.spark.ml.linalg.Vector


object EuclideanDistance extends DistanceMetric {

  override def compute(a: Vector, b: Vector): Double = {
    val x = a.toArray
    val y = b.toArray
    var i   = 0
    var sum = 0.0
    while (i < x.length) {
      val d = x(i) - y(i)
      sum += d * d
      i += 1
    }
    math.sqrt(sum)
  }
}