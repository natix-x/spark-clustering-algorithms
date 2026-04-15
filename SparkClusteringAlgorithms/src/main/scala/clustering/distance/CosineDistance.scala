package main.scala.clustering.distance


import clustering.data.Point

class CosineDistance extends DistanceMetric {

  override def compute(a: Point, b: Point): Double = {
    val dot = dotProduct(a.values, b.values)
    val normA = magnitude(a.values)
    val normB = magnitude(b.values)

    if (normA == 0.0 || normB == 0.0) {
      1.0 // maksymalna odległość (brak kierunku)
    } else {
      1.0 - (dot / (normA * normB))
    }
  }

  private def dotProduct(x: Vector[Double], y: Vector[Double]): Double = {
    var i = 0
    var sum = 0.0
    while (i < x.size) {
      sum += x(i) * y(i)
      i += 1
    }
    sum
  }

  private def magnitude(x: Vector[Double]): Double = {
    var i = 0
    var sum = 0.0
    while (i < x.size) {
      sum += x(i) * x(i)
      i += 1
    }
    math.sqrt(sum)
  }
}
