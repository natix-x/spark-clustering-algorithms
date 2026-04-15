package main.scala.clustering.distance

import clustering.data.Point


class EuclideanDistance extends DistanceMetric {

  override def compute(a: Point, b: Point): Double = {
    var i = 0
    var sum = 0.0
    while (i < a.values.size) {
      val d = a.values(i) - b.values(i)
      sum += d * d
      i += 1
    }
    math.sqrt(sum)
  }
}
