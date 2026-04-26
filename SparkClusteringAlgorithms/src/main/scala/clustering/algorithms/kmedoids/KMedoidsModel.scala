package clustering.algorithms.kmedoids

import clustering.core.Model
import clustering.data.Point
import clustering.distance.DistanceMetric
import org.apache.spark.rdd.RDD


class KMedoidsModel(
  val medoids: Array[Point],
  val distance: DistanceMetric
) extends Model {

  def predict(point: Point): Int = {
    var bestIdx = 0
    var minD    = Double.MaxValue
    var j       = 0
    while (j < medoids.length) {
      val d = distance.compute(point, medoids(j))
      if (d < minD) { minD = d; bestIdx = j }
      j += 1
    }
    bestIdx
  }

  override def labeledData(data: RDD[Point]): RDD[(Point, Int)] = {
    val bc   = data.sparkContext.broadcast(medoids)
    val dist = distance
    data.mapPartitions { iter =>
      val localMedoids = bc.value
      iter.map { p =>
        var bestIdx = 0
        var minD    = Double.MaxValue
        var j       = 0
        while (j < localMedoids.length) {
          val d = dist.compute(p, localMedoids(j))
          if (d < minD) { minD = d; bestIdx = j }
          j += 1
        }
        (p, bestIdx)
      }
    }
  }
}
