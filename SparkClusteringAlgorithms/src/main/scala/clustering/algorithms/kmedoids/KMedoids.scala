package clustering.algorithms.kmedoids

import clustering.core.Clusterer
import clustering.data.{DatasetOps, Point}
import clustering.distance.{DistanceMetric, EuclideanDistance}
import clustering.utils.MathUtils
import org.apache.spark.rdd.RDD


/** Naive K-Medoids via alternating assignment and medoid update.
 *  O(k * n²) per iteration — collected to driver.
 *  Serves as a simple baseline for comparison with PAM/CLARA.
 */
class KMedoids(
  val k: Int,
  val maxIter: Int = 100,
  val distance: DistanceMetric = new EuclideanDistance()
) extends Clusterer {

  override def fit(data: RDD[Point]): KMedoidsModel = {
    val points = DatasetOps.cachePoints(data).collect()
    val n      = points.length
    val rand   = new scala.util.Random(42)

    var medoidIndices = rand.shuffle((0 until n).toList).take(k).toArray
    var iter          = 0
    var changed       = true

    while (changed && iter < maxIter) {
      val assignments    = assignPoints(points, medoidIndices)
      val newMedoids     = updateMedoids(points, assignments)
      changed            = !newMedoids.sameElements(medoidIndices)
      medoidIndices      = newMedoids
      iter              += 1
    }

    new KMedoidsModel(medoidIndices.map(points), distance)
  }

  private def assignPoints(points: Array[Point], medoidIndices: Array[Int]): Array[Int] =
    points.map { p =>
      MathUtils.argmin(medoidIndices)(mi => distance.compute(points(mi), p))
    }

  private def updateMedoids(points: Array[Point], assignments: Array[Int]): Array[Int] =
    (0 until k).map { ci =>
      val cluster = assignments.zipWithIndex.collect { case (c, i) if c == ci => i }
      if (cluster.isEmpty) -1
      else cluster.minBy { i =>
        cluster.map(j => distance.compute(points(i), points(j))).sum
      }
    }.toArray
}
