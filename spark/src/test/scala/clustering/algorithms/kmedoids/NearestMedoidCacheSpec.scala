package clustering.algorithms.kmedoids

import clustering.algorithms.kmedoids.components.{DistanceMatrix, NearestMedoidCache}
import clustering.distance.EuclideanDistance
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.scalatest.funsuite.AnyFunSuite

import scala.util.Random

/** [[NearestMedoidCache.updateAfterSwap]] must produce EXACTLY the state a full `refresh` would,
 *  for every point, every time — it is a performance change (O(n) + a rescan of the affected
 *  points, instead of O(n·k) always), never an approximate one. Mirrors the Flink side's
 *  `NearestMedoidCacheSpec`, whose adversary (many random swaps over several k) is the same:
 *  a hand-picked example can dodge every branch of the three-way split the update does, a
 *  thousand random ones cannot.
 */
class NearestMedoidCacheSpec extends AnyFunSuite {

  private def randomPoints(n: Int, dims: Int, seed: Long): Array[Vector] = {
    val rng = new Random(seed)
    Array.fill(n)(Vectors.dense(Array.fill(dims)(rng.nextGaussian())))
  }

  private def assertSameState(expected: NearestMedoidCache, actual: NearestMedoidCache, where: String): Unit = {
    assert(expected.nearestDistance.zip(actual.nearestDistance).forall { case (a, b) => math.abs(a - b) < 1e-9 },
      s"$where: nearestDistance differs")
    assert(expected.secondNearestDistance.zip(actual.secondNearestDistance).forall { case (a, b) => math.abs(a - b) < 1e-9 },
      s"$where: secondNearestDistance differs")
    assert(expected.nearestSlot.sameElements(actual.nearestSlot), s"$where: nearestSlot differs")
    assert(expected.secondSlot.sameElements(actual.secondSlot), s"$where: secondSlot differs")
  }

  test("updateAfterSwap matches a full refresh across many random swaps") {
    val n = 300
    val points = randomPoints(n, 4, 1L)
    val distances = DistanceMatrix.pairwise(points, EuclideanDistance)

    for (k <- Seq(1, 2, 5, 20)) {
      val rng = new Random(100L + k)
      val isMedoid = new Array[Boolean](n)
      val medoids = new Array[Int](k)
      var slot = 0
      while (slot < k) {
        var candidate = rng.nextInt(n)
        while (isMedoid(candidate)) candidate = rng.nextInt(n)
        medoids(slot) = candidate
        isMedoid(candidate) = true
        slot += 1
      }

      val incremental = new NearestMedoidCache(n)
      incremental.refresh(distances, medoids)

      for (swapNum <- 0 until 500) {
        val chosenSlot = rng.nextInt(k)
        var newMedoid = rng.nextInt(n)
        while (isMedoid(newMedoid)) newMedoid = rng.nextInt(n)

        isMedoid(medoids(chosenSlot)) = false
        medoids(chosenSlot) = newMedoid
        isMedoid(newMedoid) = true

        incremental.updateAfterSwap(distances, medoids, chosenSlot)

        val fromScratch = new NearestMedoidCache(n)
        fromScratch.refresh(distances, medoids)

        assertSameState(fromScratch, incremental, s"k=$k swap#$swapNum")
      }
    }
  }

  test("updateAfterSwap handles a single medoid correctly") {
    val points = randomPoints(50, 3, 7L)
    val distances = DistanceMatrix.pairwise(points, EuclideanDistance)
    val medoids = Array(0)

    val cache = new NearestMedoidCache(50)
    cache.refresh(distances, medoids)

    medoids(0) = 17
    cache.updateAfterSwap(distances, medoids, 0)

    val expected = new NearestMedoidCache(50)
    expected.refresh(distances, medoids)

    assertSameState(expected, cache, "k=1")
  }
}
