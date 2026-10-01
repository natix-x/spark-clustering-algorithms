package clustering.algorithms.kmedoids

import clustering.algorithms.kmedoids.components.DistanceMatrix
import clustering.distance.EuclideanDistance
import org.apache.spark.ml.linalg.Vectors
import org.scalatest.funsuite.AnyFunSuite

/** Guards against the two O(n^2) failure modes `DistanceMatrix.pairwise` can hit before ever
 *  reaching `new Array[Double](...)`: a silent 32-bit overflow of `pointCount * pointCount`
 *  (previously surfaced as an opaque `NegativeArraySizeException`, e.g. `coreset`'s
 *  `inner: medoids` at m=50,000/m=100,000 in production) and a byte size that simply will not
 *  fit the driver's heap. Both must fail fast with a message naming the real cause, not crash
 *  deep inside the allocation.
 */
class DistanceMatrixSpec extends AnyFunSuite {

  private def points(n: Int) = Array.fill(n)(Vectors.dense(1.0, 2.0, 3.0))

  test("a small point count builds the matrix normally") {
    val distances = DistanceMatrix.pairwise(points(100), EuclideanDistance)
    assert(distances.pointCount === 100)
    assert(distances(0, 1) === distances(1, 0))
  }

  test("a point count whose square overflows Int.MaxValue is rejected, not silently wrapped") {
    // 46341^2 = 2,147,488,281 > Int.MaxValue (2,147,483,647); 46340^2 fits.
    val ex = intercept[IllegalArgumentException] {
      DistanceMatrix.pairwise(points(46341), EuclideanDistance)
    }
    assert(ex.getMessage.contains("capped"))
  }

  test("a point count within the Int range but exceeding the driver's memory budget is rejected") {
    // Heap-adaptive: pick n whose dense n x n matrix is just over the 0.5-of-heap budget (1.2x
    // it), while staying under the Int.MaxValue cell-count ceiling asserted above, so this test
    // is independent of how much memory the CI/dev machine happens to have.
    val maxHeapBytes = Runtime.getRuntime.maxMemory()
    val targetBytes  = (maxHeapBytes * 1.2).toLong
    val n            = Math.sqrt(targetBytes / 8.0).toInt
    assume(n.toLong * n.toLong <= Int.MaxValue.toLong, "test heap too small to stay under the Int cap")
    val ex = intercept[IllegalArgumentException] {
      DistanceMatrix.pairwise(points(n), EuclideanDistance)
    }
    assert(ex.getMessage.contains("safe budget"))
  }
}
