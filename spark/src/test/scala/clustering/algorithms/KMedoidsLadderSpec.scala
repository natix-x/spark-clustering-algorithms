package clustering.algorithms

import clustering.algorithms.kmedoids.local.{FastPAM, FasterPAM}
import clustering.distance.EuclideanDistance
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.scalatest.funsuite.AnyFunSuite

import scala.util.Random

/** Driver-local checks on the k-medoids ladder. No SparkSession — every entry has a
 *  `fitLocal` path, which is exactly the path CLARA and the sampled runs use.
 *
 *  The point of the ladder is that FastPAM and FasterPAM optimise the SAME
 *  objective with progressively cheaper search, so they must land on comparable
 *  solutions; a variant that quietly optimises something else would show up here.
 */
class KMedoidsLadderSpec extends AnyFunSuite {

  /** Three well-separated 2D blobs, 60 points each, fixed seed. */
  private val blobCentres = Array((0.0, 0.0), (50.0, 0.0), (0.0, 50.0))

  private val (points, blobOf): (Array[Vector], Array[Int]) = {
    val rnd = new Random(7L)
    val ps  = Array.newBuilder[Vector]
    val ids = Array.newBuilder[Int]
    blobCentres.zipWithIndex.foreach { case ((cx, cy), b) =>
      (0 until 60).foreach { _ =>
        ps  += Vectors.dense(cx + rnd.nextGaussian(), cy + rnd.nextGaussian())
        ids += b
      }
    }
    (ps.result(), ids.result())
  }

  private def cost(medoids: Array[Vector]): Double =
    points.map(p => medoids.map(m => EuclideanDistance.compute(p, m)).min).sum

  private def blobIndexOf(v: Vector): Int = {
    val i = points.indexWhere(p => p == v)
    assert(i >= 0, "medoid must be one of the input points")
    blobOf(i)
  }

  test("FastPAM recovers one medoid per blob") {
    val medoids = new FastPAM(k = 3, distance = EuclideanDistance).fitLocal(points).medoids
    assert(medoids.length == 3)
    assert(medoids.map(blobIndexOf).toSet == Set(0, 1, 2),
      "the exact rung must separate three well-separated blobs")
  }

  test("FasterPAM recovers one medoid per blob") {
    val medoids = new FasterPAM(k = 3, distance = EuclideanDistance).fitLocal(points).medoids
    assert(medoids.length == 3)
    assert(medoids.map(blobIndexOf).toSet == Set(0, 1, 2),
      "eager swapping must still separate three well-separated blobs")
  }

  test("FastPAM and FasterPAM agree on the same objective") {
    val fastPam   = new FastPAM(k = 3, distance = EuclideanDistance).fitLocal(points).medoids
    val fasterPam = new FasterPAM(k = 3, distance = EuclideanDistance).fitLocal(points).medoids

    // Same objective, two different searches: FastPAM takes the best swap per iteration,
    // FasterPAM the first improving one. On separated blobs both must still reach the same
    // local optimum, so the costs coincide up to floating-point noise. Equal COST is the right
    // assertion for the eager variant — a different equally-good optimum is a legitimate
    // outcome for it, unlike for FastPAM, whose search is PAM's exactly.
    assert(math.abs(cost(fasterPam) - cost(fastPam)) < 1e-9,
      s"FasterPAM cost ${cost(fasterPam)} != FastPAM cost ${cost(fastPam)}")
  }

  test("FasterPAM is deterministic for a fixed seed") {
    def run(seed: Long) = new FasterPAM(k = 3, distance = EuclideanDistance, seed = seed)
      .fitLocal(points).medoids.map(_.toString).toSeq

    assert(run(42L) == run(42L), "same seed must give the same medoids in the same order")
  }
}
