package clustering.algorithms.kmedoids

import clustering.core.Model
import clustering.distance.DistanceMetric
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}


class KMedoidsModel(
  val medoids: Array[Vector],
  val distance: DistanceMetric
) extends Model {

  def predict(features: Vector): Int = {
    var bestIdx = 0
    var minD    = Double.MaxValue
    var j       = 0
    while (j < medoids.length) {
      val d = distance.compute(features, medoids(j))
      if (d < minD) { minD = d; bestIdx = j }
      j += 1
    }
    bestIdx
  }

  override def labeledData(data: DataFrame): DataFrame = {
    val bc   = data.sparkSession.sparkContext.broadcast(medoids)
    val dist = distance
    val predictUDF = udf { features: Vector =>
      val localMedoids = bc.value
      var bestIdx = 0
      var minD    = Double.MaxValue
      var j       = 0
      while (j < localMedoids.length) {
        val d = dist.compute(features, localMedoids(j))
        if (d < minD) { minD = d; bestIdx = j }
        j += 1
      }
      bestIdx
    }
    data.withColumn("prediction", predictUDF(col("features")))
  }
}