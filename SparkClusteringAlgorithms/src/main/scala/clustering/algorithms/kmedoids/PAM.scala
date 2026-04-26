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
  val distance: DistanceMetric = EuclideanDistance
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

  private def precomputeDistances(points: Array[Point], n: Int): Array[Array[Double]] = {
    val dist = Array.ofDim[Double](n, n)
    var i = 0
    while (i < n) {
      var j = i + 1
      while (j < n) {
        val d = distance.compute(points(i), points(j))
        dist(i)(j) = d
        dist(j)(i) = d
        j += 1
      }
      i += 1
    }
    dist
  }

  private def buildPhase(dist: Array[Array[Double]], n: Int): Array[Int] = {
    val selected = new scala.collection.mutable.ArrayBuffer[Int]()

    var bestFirst = 0
    var bestSum   = Double.MaxValue
    var i = 0
    while (i < n) {
      var s = 0.0
      var j = 0
      while (j < n) { s += dist(i)(j); j += 1 }
      if (s < bestSum) { bestSum = s; bestFirst = i }
      i += 1
    }
    selected += bestFirst

    while (selected.size < k) {
      val selArr   = selected.toArray
      val nonSel   = (0 until n).filterNot(selected.contains).toArray
      var bestH    = nonSel(0)
      var bestGain = Double.MinValue

      var hi = 0
      while (hi < nonSel.length) {
        val h    = nonSel(hi)
        var gain = 0.0
        var ji   = 0
        while (ji < nonSel.length) {
          val j    = nonSel(ji)
          var dBest = Double.MaxValue
          var si   = 0
          while (si < selArr.length) {
            val d = dist(selArr(si))(j)
            if (d < dBest) dBest = d
            si += 1
          }
          val g = dBest - dist(h)(j)
          if (g > 0.0) gain += g
          ji += 1
        }
        if (gain > bestGain) { bestGain = gain; bestH = h }
        hi += 1
      }
      selected += bestH
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

    var mi = 0
    while (mi < medoids.length) {
      var hi = 0
      while (hi < nonMedoids.length) {
        val candidate = medoids.clone()
        candidate(mi) = nonMedoids(hi)
        val cost = totalCost(dist, candidate)
        if (cost < bestCost) { bestCost = cost; bestConfig = candidate }
        hi += 1
      }
      mi += 1
    }

    (bestConfig, !bestConfig.sameElements(medoids))
  }

  private def totalCost(dist: Array[Array[Double]], medoids: Array[Int]): Double = {
    var total = 0.0
    var j     = 0
    while (j < dist.length) {
      var minD = Double.MaxValue
      var mi   = 0
      while (mi < medoids.length) {
        val d = dist(medoids(mi))(j)
        if (d < minD) minD = d
        mi += 1
      }
      total += minD
      j += 1
    }
    total
  }
}
