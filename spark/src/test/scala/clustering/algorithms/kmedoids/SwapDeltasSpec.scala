package clustering.algorithms.kmedoids

import clustering.algorithms.kmedoids.local.{FastPAM, FasterPAM}
import clustering.algorithms.kmedoids.components.{DistanceMatrix, MedoidBuildPhase, NearestMedoidCache, SwapDeltas}
import clustering.distance.EuclideanDistance
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.scalatest.funsuite.AnyFunSuite

import scala.util.Random

/** The FastPAM1 Δ formula against the definition it replaces.
 *
 *  [[FastPAM]] and [[FasterPAM]] both drive their whole search off [[SwapDeltas]], so a wrong Δ
 *  would not throw — it would silently return a worse medoid set than an exhaustive search, with the ladder's
 *  "same objective" assertion as the only symptom. Here Δ is checked directly against the cost
 *  difference of the two configurations, for EVERY (slot, candidate) pair, weighted and unweighted.
 */
class SwapDeltasSpec extends AnyFunSuite {

  private val random = new Random(3L)

  private val points: Array[Vector] =
    Array.fill(80)(Vectors.dense(random.nextGaussian() * 5, random.nextGaussian() * 5, random.nextGaussian()))

  /** Σ_j w_j · min over medoids d(j, medoid) — straight from the definition. */
  private def cost(medoids: Array[Int], weights: Array[Double]): Double =
    points.indices.map { j =>
      weights(j) * medoids.map(m => EuclideanDistance.compute(points(j), points(m))).min
    }.sum

  private def checkAllDeltas(weights: Array[Double]): Unit = {
    val k         = 4
    val distances = DistanceMatrix.pairwise(points, EuclideanDistance)
    val medoids   = MedoidBuildPhase.selectInitialMedoids(distances, k, weights)
    val cache     = new NearestMedoidCache(points.length)
    cache.refresh(distances, medoids)

    val currentCost  = cost(medoids, weights)
    val deltasBySlot = new Array[Double](k)

    points.indices.filterNot(medoids.contains).foreach { candidate =>
      SwapDeltas.forCandidate(distances, candidate, weights, cache, deltasBySlot)
      (0 until k).foreach { slot =>
        val swapped = medoids.clone()
        swapped(slot) = candidate
        val expected = cost(swapped, weights) - currentCost
        assert(math.abs(deltasBySlot(slot) - expected) < 1e-9,
          s"Δ(candidate=$candidate, slot=$slot) = ${deltasBySlot(slot)} but the cost difference is $expected")
      }
    }
  }

  test("Δ equals the exact cost difference for every swap (unit weights)") {
    checkAllDeltas(Array.fill(points.length)(1.0))
  }

  test("Δ equals the exact cost difference for every swap (non-uniform weights)") {
    checkAllDeltas(Array.tabulate(points.length)(i => 1.0 + i % 7))
  }

  test("the build phase returns k distinct medoids") {
    val weights   = Array.fill(points.length)(1.0)
    val distances = DistanceMatrix.pairwise(points, EuclideanDistance)
    val medoids   = MedoidBuildPhase.selectInitialMedoids(distances, 6, weights)
    assert(medoids.length == 6)
    assert(medoids.distinct.length == 6, s"repeated medoid: ${medoids.mkString(",")}")
  }

  test("the distance matrix is symmetric and agrees with the metric") {
    val distances = DistanceMatrix.pairwise(points, EuclideanDistance)
    points.indices.foreach { i =>
      points.indices.foreach { j =>
        assert(math.abs(distances(i, j) - distances(j, i)) < 1e-12)
        assert(math.abs(distances(i, j) - EuclideanDistance.compute(points(i), points(j))) < 1e-12)
      }
    }
  }
}
