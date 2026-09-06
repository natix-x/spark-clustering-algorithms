package clustering.distance

import org.apache.spark.ml.linalg.{Vector, Vectors}

object CosineDistance extends DistanceMetric {

  override def compute(a: Vector, b: Vector): Double = {
    val normA = Vectors.norm(a, 2.0)
    val normB = Vectors.norm(b, 2.0)

    if (normA == 0.0 || normB == 0.0) {
      1.0
    } else {
      1.0 - (a.dot(b) / (normA * normB))
    }
  }

  /** Raw-array full distance: dot product and both norms in ONE pass over the coordinates,
   *  where the `Vector` form makes three (dot, norm a, norm b). */
  override def compute(a: Array[Double], b: Array[Double]): Double = {
    var dot = 0.0
    var sumA = 0.0
    var sumB = 0.0
    var i = 0
    while (i < a.length) {
      dot += a(i) * b(i)
      sumA += a(i) * a(i)
      sumB += b(i) * b(i)
      i += 1
    }
    if (sumA == 0.0 || sumB == 0.0) 1.0
    else 1.0 - (dot / (math.sqrt(sumA) * math.sqrt(sumB)))
  }

  /** No early exit is possible (the dot product can move either way until the last coordinate),
   *  but the raw loop still avoids the wrapper and computes both norms in the same pass. */
  override def withinRadius(a: Array[Double], b: Array[Double], radius: Double): Boolean = {
    var dot = 0.0
    var sumA = 0.0
    var sumB = 0.0
    var i = 0
    while (i < a.length) {
      dot += a(i) * b(i)
      sumA += a(i) * a(i)
      sumB += b(i) * b(i)
      i += 1
    }
    if (sumA == 0.0 || sumB == 0.0) 1.0 <= radius
    else 1.0 - (dot / (math.sqrt(sumA) * math.sqrt(sumB))) <= radius
  }

  /** No early exit here either — the dot product can move either way until the last coordinate —
   *  so this is the raw-array loop plus the bound test. Still worth overriding: it skips the
   *  `Vectors.dense` wrappers and the two separate norm passes of the default. */
  override def distanceUpTo(a: Array[Double], b: Array[Double], bound: Double): Double = {
    var dot = 0.0
    var sumA = 0.0
    var sumB = 0.0
    var i = 0
    while (i < a.length) {
      dot += a(i) * b(i)
      sumA += a(i) * a(i)
      sumB += b(i) * b(i)
      i += 1
    }
    val d = if (sumA == 0.0 || sumB == 0.0) 1.0 else 1.0 - (dot / (math.sqrt(sumA) * math.sqrt(sumB)))
    if (d <= bound) d else Double.PositiveInfinity
  }
}
