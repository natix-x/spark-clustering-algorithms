package clustering.distance

import org.apache.spark.ml.linalg.Vector


object CosineDistance extends DistanceMetric {

  override def compute(a: Vector, b: Vector): Double = {
    val x     = a.toArray
    val y     = b.toArray
    val dot   = dotProduct(x, y)
    val normA = magnitude(x)
    val normB = magnitude(y)

    if (normA == 0.0 || normB == 0.0) {
      1.0
    } else {
      1.0 - (dot / (normA * normB))
    }
  }

  private def dotProduct(x: Array[Double], y: Array[Double]): Double = {
    var i   = 0
    var sum = 0.0
    while (i < x.length) {
      sum += x(i) * y(i)
      i += 1
    }
    sum
  }

  private def magnitude(x: Array[Double]): Double = {
    var i   = 0
    var sum = 0.0
    while (i < x.length) {
      sum += x(i) * x(i)
      i += 1
    }
    math.sqrt(sum)
  }
}
