package main.scala.clustering.distance

import clustering.data.Point


class ManhattanDistance extends DistanceMetric {

  override def compute(a: Point, b: Point): Double = {
    var i = 0
    var sum = 0.0
    while (i < a.values.size) {
      sum += math.abs(a.values(i) - b.values(i))
      i += 1
    }
    sum
  }
}
