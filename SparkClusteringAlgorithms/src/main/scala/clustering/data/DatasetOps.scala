package clustering.data

import org.apache.spark.rdd.RDD


object DatasetOps {

  def cachePoints(data: RDD[Point]): RDD[Point] =
    data.cache()

  def computeCentroid(points: Iterable[Point]): Point = {
    val dim = points.head.values.size
    val sum = Array.fill(dim)(0.0)

    points.foreach { p =>
      p.values.zipWithIndex.foreach { case (v, i) =>
        sum(i) += v
      }
    }

    val n = points.size.toDouble
    Point(sum.map(_ / n).toVector)
  }
}
