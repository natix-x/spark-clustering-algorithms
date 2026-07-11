package clustering.algorithms.kmeans

import clustering.core.Model
import clustering.distance.DistanceMetric
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}


class KMeansModel(
  val centroids: Array[Vector],
  val distance: DistanceMetric
) extends Model {

  def predict(features: Vector): Int = {
    var bestIdx = 0
    var minD    = Double.MaxValue
    var j       = 0
    while (j < centroids.length) {
      val d = distance.compute(features, centroids(j))
      if (d < minD) { minD = d; bestIdx = j }
      j += 1
    }
    bestIdx
  }

  /** Broadcasts the centroids once and assigns labels via a UDF, instead of
   *  serialising the model into every task closure. */
  override def labeledData(data: DataFrame): DataFrame = {
    val bc   = data.sparkSession.sparkContext.broadcast(centroids)
    val dist = distance
    val predictUDF = udf { features: Vector =>
      val localCentroids = bc.value
      var bestIdx = 0
      var minD    = Double.MaxValue
      var j       = 0
      while (j < localCentroids.length) {
        val d = dist.compute(features, localCentroids(j))
        if (d < minD) { minD = d; bestIdx = j }
        j += 1
      }
      bestIdx
    }
    data.withColumn("prediction", predictUDF(col("features")))
  }
}