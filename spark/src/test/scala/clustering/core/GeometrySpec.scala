package clustering.core

import clustering.benchmark.registry.GeometryRegistry
import clustering.distance.{CosineDistance, UnitSphereDistance}
import org.apache.spark.ml.linalg.Vectors
import org.scalatest.funsuite.AnyFunSuite

/** Guards the two invariants the spherical geometry rests on: normalisation really
 *  produces unit vectors, and `1 − dot` really equals cosine distance there — the
 *  substitution that makes spherical k-means cheaper per iteration. */
class GeometrySpec extends AnyFunSuite {

  test("l2Normalise produces unit vectors and leaves the zero vector alone") {
    val v = Vectors.dense(3.0, 4.0)
    val u = Geometry.l2Normalize(v)
    assert(math.abs(math.sqrt(u.toArray.map(x => x * x).sum) - 1.0) < 1e-12)
    assert(u.toArray.sameElements(Array(0.6, 0.8)))

    val zero = Vectors.dense(0.0, 0.0)
    assert(Geometry.l2Normalize(zero) == zero)
  }

  test("UnitSphereDistance equals CosineDistance on normalised vectors") {
    val pairs = Seq(
      (Vectors.dense(1.0, 2.0, 3.0), Vectors.dense(-2.0, 0.5, 4.0)),
      (Vectors.dense(5.0, 0.0),      Vectors.dense(0.0, 7.0)),
      (Vectors.dense(1.0, 1.0),      Vectors.dense(2.0, 2.0))
    )
    pairs.foreach { case (a, b) =>
      val (ua, ub) = (Geometry.l2Normalize(a), Geometry.l2Normalize(b))
      assert(math.abs(UnitSphereDistance.compute(ua, ub) - CosineDistance.compute(ua, ub)) < 1e-12)
    }
  }

  test("spherical geometry's fitDistance is always UnitSphereDistance") {
    assert(SphericalGeometry.fitDistance == UnitSphereDistance)
  }

  test("spherical projection keeps centroids on the unit sphere, euclidean is the identity") {
    val mean = Vectors.dense(0.1, 0.2, 0.3)
    assert(math.abs(SphericalGeometry.project(mean).toArray.map(x => x * x).sum - 1.0) < 1e-12)
    assert(EuclideanGeometry.project(mean) == mean)
  }

  /** `pointCost` has to be the objective the geometry's own iteration optimises, because every
   *  criterion layered on the loop ranks by it — bisecting's leaf and trial choice, breathing's
   *  error and utility. Lloyd minimises Σ d², spherical k-means (Dhillon & Modha) maximises
   *  Σ cos, i.e. minimises Σ (1 − cos) unsquared. */
  test("point cost is squared under euclidean and linear under spherical") {
    assert(EuclideanGeometry.pointCost(3.0) == 9.0)
    assert(SphericalGeometry.pointCost(0.25) == 0.25)
  }

  test("costToDistance inverts pointCost, so an offset scale stays a distance") {
    Seq(0.0, 0.25, 1.0, 7.5).foreach { d =>
      assert(math.abs(EuclideanGeometry.costToDistance(EuclideanGeometry.pointCost(d)) - d) < 1e-12)
      assert(math.abs(SphericalGeometry.costToDistance(SphericalGeometry.pointCost(d)) - d) < 1e-12)
    }
  }

  /** The identity that makes the spherical choice reproduce Steinbach, Karypis & Kumar's split
   *  criterion exactly: they score a cluster by average pairwise similarity, which for unit
   *  vectors is ‖c‖², and Σ (1 − cos) = n(1 − ‖c‖) is monotone in that. The squared form is not,
   *  which is why `pointCost` is a geometry hook rather than a fixed d². */
  test("spherical point cost sums to n(1 − ‖c‖) for unit vectors") {
    val points = Seq(
      Vectors.dense(1.0, 0.0), Vectors.dense(0.0, 1.0),
      Vectors.dense(1.0, 1.0), Vectors.dense(3.0, 1.0)
    ).map(Geometry.l2Normalize)

    val mean     = Vectors.dense(points.map(_.toArray).transpose.map(_.sum / points.size).toArray)
    val meanNorm = math.sqrt(mean.toArray.map(x => x * x).sum)
    val centroid = SphericalGeometry.project(mean)

    val summed = points.map(p => SphericalGeometry.pointCost(UnitSphereDistance.compute(p, centroid))).sum
    assert(math.abs(summed - points.size * (1.0 - meanNorm)) < 1e-12)
  }

  test("registry resolves both geometries and rejects unknown ones") {
    assert(GeometryRegistry.get("euclidean") == EuclideanGeometry)
    assert(GeometryRegistry.get("SPHERICAL") == SphericalGeometry)
    assert(GeometryRegistry.Default == EuclideanGeometry)
    intercept[IllegalArgumentException](GeometryRegistry.get("hyperbolic"))
  }
}
