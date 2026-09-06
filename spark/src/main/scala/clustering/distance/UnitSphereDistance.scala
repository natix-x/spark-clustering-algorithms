package clustering.distance

import org.apache.spark.ml.linalg.Vector

/** Cosine distance specialised to **unit-norm** vectors: `1 − ⟨a, b⟩`.
 *
 *  Identical to [[CosineDistance]] whenever both arguments are L2-normalised, but
 *  it skips the two `sqrt(Σ x²)` passes — which is exactly the per-iteration saving
 *  spherical k-means is supposed to deliver at high dimensionality.
 *
 *  Deliberately NOT registered in `DistanceRegistry`: it is only correct on
 *  normalised data, so it may not be selected from a config. It is chosen
 *  internally by `SphericalGeometry`, which guarantees the normalisation.
 */

object UnitSphereDistance extends DistanceMetric {
  override def compute(a: Vector, b: Vector): Double = {
    1.0 - a.dot(b)
  }

  override def compute(a: Array[Double], b: Array[Double]): Double = {
    var dot = 0.0
    var i = 0
    while (i < a.length) {
      dot += a(i) * b(i)
      i += 1
    }
    1.0 - dot
  }

  override def withinRadius(a: Array[Double], b: Array[Double], radius: Double): Boolean = {
    var dot = 0.0
    var i = 0
    while (i < a.length) {
      dot += a(i) * b(i)
      i += 1
    }
    1.0 - dot <= radius
  }

  override def distanceUpTo(a: Array[Double], b: Array[Double], bound: Double): Double = {
    var dot = 0.0
    var i = 0
    while (i < a.length) {
      dot += a(i) * b(i)
      i += 1
    }
    val d = 1.0 - dot
    if (d <= bound) d else Double.PositiveInfinity
  }
}
