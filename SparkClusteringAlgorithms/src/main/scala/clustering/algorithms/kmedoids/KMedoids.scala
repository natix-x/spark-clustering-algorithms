package clustering.algorithms.kmedoids

import clustering.core.Clusterer
import clustering.data.{DatasetOps, Point}
import clustering.distance.{DistanceMetric, EuclideanDistance}
import org.apache.spark.rdd.RDD


/** Naive K-Medoids via alternating assignment and medoid update.
 *  O(k * n²) per iteration — collected to driver.
 *  Serves as a simple baseline for comparison with PAM/CLARA.
 */
class KMedoids(
  val k: Int,
  val maxIter: Int = 100,
  val distance: DistanceMetric = EuclideanDistance
) extends Clusterer {

  override def fit(data: RDD[Point]): KMedoidsModel = {
    val points = DatasetOps.cachePoints(data).collect()
    val n      = points.length
    val rand   = new scala.util.Random(42)

    var medoidIndices = rand.shuffle((0 until n).toList).take(k).toArray
    var iter          = 0
    var changed       = true

    while (changed && iter < maxIter) {
      val assignments = assignPoints(points, medoidIndices)
      val newMedoids  = updateMedoids(points, assignments)
      changed         = !newMedoids.sameElements(medoidIndices)
      medoidIndices   = newMedoids
      iter           += 1
    }

    new KMedoidsModel(medoidIndices.map(points), distance)
  }

  private def assignPoints(points: Array[Point], medoidIndices: Array[Int]): Array[Int] = {
    val assignments = new Array[Int](points.length)
    var i = 0
    while (i < points.length) {
      var bestIdx = 0
      var minD    = Double.MaxValue
      var j       = 0
      while (j < medoidIndices.length) {
        val d = distance.compute(points(i), points(medoidIndices(j)))
        if (d < minD) { minD = d; bestIdx = j }
        j += 1
      }
      assignments(i) = bestIdx
      i += 1
    }
    assignments
  }

  private def updateMedoids(points: Array[Point], assignments: Array[Int]): Array[Int] = {
    val newMedoids = new Array[Int](k)
    var ci = 0
    while (ci < k) {
      var clusterSize = 0
      var idx = 0
      while (idx < assignments.length) {
        if (assignments(idx) == ci) clusterSize += 1
        idx += 1
      }
      val cluster = new Array[Int](clusterSize)
      var pos = 0
      idx = 0
      while (idx < assignments.length) {
        if (assignments(idx) == ci) { cluster(pos) = idx; pos += 1 }
        idx += 1
      }
      if (cluster.isEmpty) {
        newMedoids(ci) = -1
      } else {
        var bestI    = cluster(0)
        var bestCost = Double.MaxValue
        var ii = 0
        while (ii < cluster.length) {
          var cost = 0.0
          var jj = 0
          while (jj < cluster.length) {
            cost += distance.compute(points(cluster(ii)), points(cluster(jj)))
            jj += 1
          }
          if (cost < bestCost) { bestCost = cost; bestI = cluster(ii) }
          ii += 1
        }
        newMedoids(ci) = bestI
      }
      ci += 1
    }
    newMedoids
  }
}
