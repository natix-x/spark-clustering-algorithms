package clustering.algorithms.kmedoids

import clustering.core.Clusterer
import clustering.data.{DatasetOps, Point}
import clustering.distance.{DistanceMetric, EuclideanDistance}
import org.apache.spark.rdd.RDD


/** Partitioning Around Medoids (PAM).
 *
 *  BUILD phase: greedy O(k * n) initialisation that minimises total
 *  dissimilarity.
 *  SWAP phase: tries every (medoid, non-medoid) exchange and applies
 *  the best swap until no improvement — O(k * (n-k)) cost per iteration.
 *
 *  More accurate than naive KMedoids at the cost of higher runtime.
 *  Collects data to driver; suited for moderate dataset sizes.
 */
class PAM(
  val k: Int,
  val maxIter: Int = 100,
  val distance: DistanceMetric = new EuclideanDistance()
) extends Clusterer {

  override def fit(data: RDD[Point]): KMedoidsModel = {
    val points = DatasetOps.cachePoints(data).collect()
    val n      = points.length
    val dist   = precomputeDistances(points, n)

    var medoids  = buildPhase(dist, n)
    var iter     = 0
    var improved = true

    while (improved && iter < maxIter) {
      val (next, did) = swapPhase(dist, n, medoids)
      medoids  = next
      improved = did
      iter    += 1
    }

    new KMedoidsModel(medoids.map(points), distance)
  }

  private def precomputeDistances(points: Array[Point], n: Int): Array[Array[Double]] =
    Array.tabulate(n, n) { (i, j) =>
      if (i == j) 0.0 else distance.compute(points(i), points(j))
    }

  private def buildPhase(dist: Array[Array[Double]], n: Int): Array[Int] = {
    val selected = scala.collection.mutable.ArrayBuffer[Int]()

    selected += (0 until n).minBy(i => (0 until n).map(dist(i)(_)).sum)

    while (selected.size < k) {
      val nonSelected = (0 until n).filterNot(selected.contains)
      val next = nonSelected.maxBy { h =>
        nonSelected.map { j =>
          val dBest = selected.map(dist(_)(j)).min
          math.max(0.0, dBest - dist(h)(j))
        }.sum
      }
      selected += next
    }

    selected.toArray
  }

  private def swapPhase(
    dist: Array[Array[Double]],
    n: Int,
    medoids: Array[Int]
  ): (Array[Int], Boolean) = {
    val medoidSet  = medoids.toSet
    val nonMedoids = (0 until n).filterNot(medoidSet.contains).toArray
    var bestCost   = totalCost(dist, medoids)
    var bestConfig = medoids

    for (mi <- medoids.indices; h <- nonMedoids) {
      val candidate = medoids.clone()
      candidate(mi) = h
      val cost = totalCost(dist, candidate)
      if (cost < bestCost) {
        bestCost   = cost
        bestConfig = candidate
      }
    }

    (bestConfig, !bestConfig.sameElements(medoids))
  }

  private def totalCost(dist: Array[Array[Double]], medoids: Array[Int]): Double =
    dist.indices.map(j => medoids.map(dist(_)(j)).min).sum
}
