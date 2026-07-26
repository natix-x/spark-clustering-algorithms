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
}
