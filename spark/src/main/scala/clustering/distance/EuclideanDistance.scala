package clustering.distance

import org.apache.spark.ml.linalg.{Vector, Vectors}

object EuclideanDistance extends DistanceMetric {

  override def compute(a: Vector, b: Vector): Double = {
    math.sqrt(Vectors.sqdist(a, b))
  }

  /** Squared distances, so no `sqrt`, plus an EARLY EXIT: once the partial sum passes r² the
   *  remaining coordinates cannot bring it back. */
  override def withinRadius(a: Array[Double], b: Array[Double], radius: Double): Boolean = {
    val limit = radius * radius
    var sum = 0.0
    var i = 0
    while (i < a.length) {
      val d = a(i) - b(i)
      sum += d * d
      if (sum > limit) return false
      i += 1
    }
    sum <= limit
  }

  /** Squared-space early exit, `sqrt` paid only for a hit — and hits are the rare case in a
   *  nearest-core scan over a large core set. */
  override def distanceUpTo(a: Array[Double], b: Array[Double], bound: Double): Double = {
    val limit = bound * bound
    var sum = 0.0
    var i = 0
    while (i < a.length) {
      val d = a(i) - b(i)
      sum += d * d
      if (sum > limit) return Double.PositiveInfinity
      i += 1
    }
    if (sum <= limit) math.sqrt(sum) else Double.PositiveInfinity
  }
}
