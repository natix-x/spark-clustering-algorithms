package clustering.algorithms.kmeans

import clustering.core.Model
import clustering.data.Point
import clustering.distance.DistanceMetric
import org.apache.spark.rdd.RDD


class KMeansModel(
  val centroids: Array[Point],
  val distance: DistanceMetric
) extends Model {

  def predict(point: Point): Int = {
    var bestIdx = 0
    var minD    = Double.MaxValue
    var j       = 0
    while (j < centroids.length) {
      val d = distance.compute(point, centroids(j))
      if (d < minD) { minD = d; bestIdx = j }
      j += 1
    }
    bestIdx
  }

  override def labeledData(data: RDD[Point]): RDD[(Point, Int)] = {
    val bc = data.sparkContext.broadcast(centroids)
    val dist = distance
    data.mapPartitions { iter =>
      val localCentroids = bc.value
      iter.map { p =>
        var bestIdx = 0
        var minD    = Double.MaxValue
        var j       = 0
        while (j < localCentroids.length) {
          val d = dist.compute(p, localCentroids(j))
          if (d < minD) { minD = d; bestIdx = j }
          j += 1
        }
        (p, bestIdx)
      }
    }
  }
}
