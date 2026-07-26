package clustering.distance

import org.apache.spark.ml.linalg.Vector

// TODO: make it more performant if possible
object ManhattanDistance extends DistanceMetric {

  override def compute(a: Vector, b: Vector): Double = {
    val x = a.toArray
    val y = b.toArray
    var i   = 0
    var sum = 0.0
    while (i < x.length) {
      sum += math.abs(x(i) - y(i))
      i += 1
    }
    sum
  }
}
