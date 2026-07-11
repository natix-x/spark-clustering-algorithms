package clustering.core

import clustering.distance.DistanceMetric
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}

/** A model defined by a set of prototype vectors (centroids, medoids, …).
 *
 *  Each point is labelled with the index of its nearest prototype under
 *  `distance`. Prototypes are broadcast once per model (memoised across
 *  `assignClusters` calls) so the array is neither serialised into every task
 *  closure nor re-broadcast on each evaluation pass. The broadcast lives for
 *  the model's lifetime and is reclaimed when the SparkContext stops.
 */
abstract class NearestPrototypeModel(
  val prototypes: Array[Vector],
  val distance:   DistanceMetric
) extends Model {

  require(prototypes.nonEmpty, "NearestPrototypeModel: prototypes must be non-empty")

  @transient private var bcPrototypes: Broadcast[Array[Vector]] = _

  private def broadcast(data: DataFrame): Broadcast[Array[Vector]] = synchronized {
    if (bcPrototypes == null)
      bcPrototypes = data.sparkSession.sparkContext.broadcast(prototypes)
    bcPrototypes
  }

  override def assignClusters(data: DataFrame): DataFrame = {
    val bc         = broadcast(data)
    val dist       = distance
    val predictUDF = udf { features: Vector =>
      NearestPrototypeModel.nearest(features, bc.value, dist)
    }
    data.withColumn(Columns.Prediction, predictUDF(col(Columns.Features)))
  }
}

object NearestPrototypeModel {

  /** Index of the prototype closest to `features` under `distance`. */
  def nearest(features: Vector, prototypes: Array[Vector], distance: DistanceMetric): Int = {
    var bestIdx = 0
    var minD    = Double.MaxValue
    var j       = 0
    while (j < prototypes.length) {
      val d = distance.compute(features, prototypes(j))
      if (d < minD) { minD = d; bestIdx = j }
      j += 1
    }
    bestIdx
  }
}
