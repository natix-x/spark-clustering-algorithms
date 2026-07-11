package clustering.evaluation

import clustering.core.Model
import clustering.distance.{DistanceMetric, EuclideanDistance}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._


class SilhouetteEvaluator(
                           val distance: DistanceMetric = EuclideanDistance
                         ) extends ClusteringEvaluator {

  override def evaluate(model: Model, data: DataFrame): Double = {
    val spark = data.sparkSession

    val labeled = model.assignClusters(data)
      .filter(col("prediction") =!= -1)
      .select(col("prediction"), col("features"))
      .cache()

    // Build cluster -> points map on the driver. Silhouette is O(n²), so the
    // caller samples the data to a tractable size before evaluation.
    val clusterMap: Map[Int, Array[Vector]] = labeled.collect()
      .map(r => (r.getInt(0), r.getAs[Vector]("features")))
      .groupBy(_._1)
      .map { case (l, arr) => l -> arr.map(_._2) }

    if (clusterMap.isEmpty) {
      labeled.unpersist()
      return 0.0
    }

    val bcClusters = spark.sparkContext.broadcast(clusterMap)
    val dist       = distance

    val silUDF = udf { (label: Int, p: Vector) =>
      val clus = bcClusters.value
      val a    = SilhouetteEvaluator.meanDistExclSelf(p, clus.getOrElse(label, Array.empty[Vector]), dist)
      val b    = clus.iterator
        .filter(_._1 != label)
        .map { case (_, pts) => SilhouetteEvaluator.meanDist(p, pts, dist) }
        .reduceOption(_ min _)
        .getOrElse(Double.MaxValue)
      val denom = math.max(a, b)
      if (denom == 0.0) 0.0 else (b - a) / denom
    }

    val row = labeled
      .select(silUDF(col("prediction"), col("features")).as("s"))
      .agg(sum("s").as("scoreSum"), count("s").as("cnt"))
      .head()

    val scoreSum = if (row.isNullAt(0)) 0.0 else row.getDouble(0)
    val cnt      = row.getLong(1)

    bcClusters.destroy()
    labeled.unpersist()
    if (cnt == 0L) 0.0 else scoreSum / cnt
  }
}

object SilhouetteEvaluator {

  private def meanDist(p: Vector, pts: Array[Vector], dist: DistanceMetric): Double = {
    if (pts.isEmpty) return 0.0
    var s = 0.0
    var i = 0
    while (i < pts.length) { s += dist.compute(p, pts(i)); i += 1 }
    s / pts.length
  }

  /** Mean distance to other points in the same cluster, excluding `p` itself. */
  private def meanDistExclSelf(p: Vector, pts: Array[Vector], dist: DistanceMetric): Double = {
    var s   = 0.0
    var cnt = 0
    var i   = 0
    while (i < pts.length) {
      val q = pts(i)
      if (!q.equals(p)) { s += dist.compute(p, q); cnt += 1 }
      i += 1
    }
    if (cnt == 0) 0.0 else s / cnt
  }
}
