package clustering.distance

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.scalatest.funsuite.AnyFunSuite

import scala.util.Random

/** `withinRadius` is an optimisation of `compute(a, b) <= radius`, and `distanceUpTo` the same for
 *  a bounded nearest-neighbour scan; the ε-scans of `dbscanpp` (steps 2 and 3) and its labelling
 *  pass use nothing else — so a disagreement here is a wrong core point or a wrong label, i.e. a
 *  wrong cluster count, with no other symptom. This suite pins both to `compute` for every metric.
 *
 *  The boundary is checked explicitly: `<=` is what makes a point exactly ON the ε-sphere a
 *  neighbour, and squaring both sides must not move that.
 */
class RadiusPredicateSpec extends AnyFunSuite {

  private val metrics: Seq[(String, DistanceMetric)] = Seq(
    "euclidean"   -> EuclideanDistance,
    "manhattan"   -> ManhattanDistance,
    "cosine"      -> CosineDistance,
    "unit-sphere" -> UnitSphereDistance
  )

  test("the predicate agrees with compute over random pairs and radii") {
    val random = new Random(17L)
    metrics.foreach { case (name, metric) =>
      (0 until 400).foreach { _ =>
        val dim = 1 + random.nextInt(8)
        val a = Array.fill(dim)(random.nextDouble() * 4 - 2)
        val b = Array.fill(dim)(random.nextDouble() * 4 - 2)
        val radius = random.nextDouble() * 4
        val expected = metric.compute(Vectors.dense(a), Vectors.dense(b)) <= radius

        assert(metric.withinRadius(a, b, radius) == expected,
          s"$name: withinRadius disagreed for r=$radius")
      }
    }
  }

  test("a point exactly on the radius is inside it") {
    val a = Vectors.dense(0.0, 0.0)
    val b = Vectors.dense(3.0, 4.0)   // euclidean 5, manhattan 7
    assert(EuclideanDistance.withinRadius(a.toArray, b.toArray, 5.0))
    assert(!EuclideanDistance.withinRadius(a.toArray, b.toArray, 4.999999))
    assert(ManhattanDistance.withinRadius(a.toArray, b.toArray, 7.0))
    assert(!ManhattanDistance.withinRadius(a.toArray, b.toArray, 6.999999))
  }

  test("the early exit does not change the verdict at any dimensionality") {
    // A pair that only exceeds the radius in its LAST coordinate: exiting early on a partial sum
    // must not report "inside" before that coordinate is seen.
    val dim = 64
    val a = Array.fill(dim)(0.0)
    val b = Array.fill(dim)(0.0)
    b(dim - 1) = 10.0
    assert(!EuclideanDistance.withinRadius(a, b, 9.0))
    assert(EuclideanDistance.withinRadius(a, b, 10.0))
    assert(!ManhattanDistance.withinRadius(a, b, 9.0))
    assert(ManhattanDistance.withinRadius(a, b, 10.0))
  }

  test("NaN coordinates are outside every radius, exactly as compute says") {
    val a = Array(0.0, Double.NaN)
    val b = Array(0.0, 0.0)
    metrics.foreach { case (name, metric) =>
      val expected = metric.compute(Vectors.dense(a), Vectors.dense(b)) <= 1e9
      assert(metric.withinRadius(a, b, 1e9) == expected, s"$name: NaN handling diverged")
    }
  }

  /** Every dataset in the matrix is dense, but the scans reach the coordinates through
   *  `Vector.toArray`, which also accepts a `SparseVector` (one densified copy per row). So sparse
   *  input must give the same verdict as the equivalent dense vector — the cheap guarantee that the
   *  simplification to a single raw path did not quietly become dense-only. */
  test("sparse input gives the same verdict as its dense equivalent") {
    val sparse: Vector = Vectors.sparse(5, Array(1, 4), Array(3.0, 4.0))
    val dense: Vector = Vectors.dense(sparse.toArray)
    val other: Vector = Vectors.dense(0.0, 0.0, 0.0, 0.0, 0.0)
    Seq(0.5, 4.999999, 5.0, 7.0).foreach { radius =>
      assert(EuclideanDistance.withinRadius(sparse.toArray, other.toArray, radius) ==
             EuclideanDistance.withinRadius(dense.toArray, other.toArray, radius),
        s"sparse and dense disagreed at r=$radius")
      assert(EuclideanDistance.withinRadius(sparse.toArray, other.toArray, radius) ==
             (EuclideanDistance.compute(sparse, other) <= radius),
        s"sparse verdict left compute behind at r=$radius")
    }
  }

  // ── distanceUpTo ──────────────────────────────────────────────────────────────────────────
  // The nearest-neighbour sibling of withinRadius, used by CoreLabelModel's n·m labelling scan.
  // Same contract, one more thing to get right: when it returns a finite value that value must be
  // the true distance, because the caller keeps it as the running minimum.

  test("distanceUpTo returns the true distance below the bound and infinity above it") {
    val random = new Random(23L)
    metrics.foreach { case (name, metric) =>
      (0 until 400).foreach { _ =>
        val dim = 1 + random.nextInt(8)
        val a = Array.fill(dim)(random.nextDouble() * 4 - 2)
        val b = Array.fill(dim)(random.nextDouble() * 4 - 2)
        val bound = random.nextDouble() * 4
        val exact = metric.compute(Vectors.dense(a), Vectors.dense(b))
        val bounded = metric.distanceUpTo(a, b, bound)

        if (exact <= bound) {
          assert(math.abs(bounded - exact) < 1e-9,
            s"$name: distanceUpTo returned $bounded, compute says $exact (bound $bound)")
        } else {
          assert(bounded == Double.PositiveInfinity,
            s"$name: distanceUpTo returned $bounded for a pair beyond the bound $bound")
        }
      }
    }
  }

  test("a point exactly on the bound is a hit, and reports its distance") {
    val a = Array(0.0, 0.0)
    val b = Array(3.0, 4.0)   // euclidean 5, manhattan 7
    assert(EuclideanDistance.distanceUpTo(a, b, 5.0) == 5.0)
    assert(EuclideanDistance.distanceUpTo(a, b, 4.999999) == Double.PositiveInfinity)
    assert(ManhattanDistance.distanceUpTo(a, b, 7.0) == 7.0)
    assert(ManhattanDistance.distanceUpTo(a, b, 6.999999) == Double.PositiveInfinity)
  }

  test("distanceUpTo's early exit does not report a hit before the last coordinate is seen") {
    val dim = 64
    val a = Array.fill(dim)(0.0)
    val b = Array.fill(dim)(0.0)
    b(dim - 1) = 10.0
    assert(EuclideanDistance.distanceUpTo(a, b, 9.0) == Double.PositiveInfinity)
    assert(EuclideanDistance.distanceUpTo(a, b, 10.0) == 10.0)
    assert(ManhattanDistance.distanceUpTo(a, b, 9.0) == Double.PositiveInfinity)
    assert(ManhattanDistance.distanceUpTo(a, b, 10.0) == 10.0)
  }

  /** NaN follows `compute`, not intuition. For the L-norms a NaN coordinate poisons the sum, every
   *  comparison against the bound is false and the pair is reported as beyond it. Cosine is the
   *  exception and legitimately so: `b` here is the zero vector, for which `compute` short-circuits
   *  to 1.0 without ever looking at `a`'s coordinates — so a finite answer is the correct one, and
   *  asserting infinity for every metric would be asserting a bug. */
  test("NaN coordinates follow compute, whatever it says") {
    val a = Array(0.0, Double.NaN)
    val b = Array(0.0, 0.0)
    metrics.foreach { case (name, metric) =>
      val exact   = metric.compute(Vectors.dense(a), Vectors.dense(b))
      val bounded = metric.distanceUpTo(a, b, 1e9)
      if (exact <= 1e9) {
        assert(math.abs(bounded - exact) < 1e-9, s"$name: got $bounded, compute says $exact")
      } else {
        assert(bounded == Double.PositiveInfinity, s"$name: NaN slipped through as $bounded")
      }
    }
  }

  /** An infinite bound must behave like an unbounded scan — that is the `assign: closest` path,
   *  where no ε caps the search. */
  test("an infinite bound always reports the exact distance") {
    val a = Array(1.0, 2.0, 3.0)
    val b = Array(-1.0, 0.5, 7.0)
    metrics.foreach { case (name, metric) =>
      val exact = metric.compute(Vectors.dense(a), Vectors.dense(b))
      assert(math.abs(metric.distanceUpTo(a, b, Double.PositiveInfinity) - exact) < 1e-9,
        s"$name: an unbounded call did not match compute")
    }
  }
}