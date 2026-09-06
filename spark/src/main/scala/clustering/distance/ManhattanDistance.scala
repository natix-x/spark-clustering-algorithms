package clustering.distance

import org.apache.spark.ml.linalg.Vector

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

  /** Raw-array full distance: no wrapper, no `toArray` round-trip. */
  override def compute(a: Array[Double], b: Array[Double]): Double = {
    var sum = 0.0
    var i = 0
    while (i < a.length) {
      sum += math.abs(a(i) - b(i))
      i += 1
    }
    sum
  }

  /** Same early exit as [[EuclideanDistance.withinRadius]]: L1 sums are monotone, so once the
   *  partial sum passes the radius the pair is out. */
  override def withinRadius(a: Array[Double], b: Array[Double], radius: Double): Boolean = {
    var sum = 0.0
    var i = 0
    while (i < a.length) {
      sum += math.abs(a(i) - b(i))
      if (sum > radius) return false
      i += 1
    }
    sum <= radius
  }

  /** L1 sums are monotone, so the same early exit works against a shrinking bound. */
  override def distanceUpTo(a: Array[Double], b: Array[Double], bound: Double): Double = {
    var sum = 0.0
    var i = 0
    while (i < a.length) {
      sum += math.abs(a(i) - b(i))
      if (sum > bound) return Double.PositiveInfinity
      i += 1
    }
    if (sum <= bound) sum else Double.PositiveInfinity
  }
}
