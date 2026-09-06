package clustering.core

import clustering.distance.DistanceMetric
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}

/** A model defined by a set of prototype vectors (centroids, medoids, etc.).
 *
 *  Each point is assigned to its nearest prototype. Prototypes are broadcast
 *  as raw arrays on the first `assignClusters` call and memoised, avoiding
 *  serialization overhead on subsequent evaluations.
 */
abstract class NearestPrototypeModel(
  val prototypes: Array[Vector],
  val distance: DistanceMetric
) extends Model {

  require(prototypes.nonEmpty, "NearestPrototypeModel: prototypes must be non-empty")

  @transient private var bcPrototypes: Broadcast[Array[Array[Double]]] = _

  private def getBroadcast(data: DataFrame): Broadcast[Array[Array[Double]]] = synchronized {
    if (bcPrototypes == null) {
      bcPrototypes = data.sparkSession.sparkContext.broadcast(prototypes.map(_.toArray))
    }
    bcPrototypes
  }

  override def assignClusters(data: DataFrame): DataFrame = {
    val bc = getBroadcast(data)
    val dist = distance // Local reference for closure serialization

    val predictUDF = udf { features: Vector =>
      NearestPrototypeModel.nearestRaw(features.toArray, bc.value, dist)
    }

    data.withColumn(Columns.Prediction, predictUDF(col(Columns.Features)))
  }
}

object NearestPrototypeModel {

  /** Index of the prototype closest to `features` under `distance`. */
  def nearest(features: Vector, prototypes: Array[Vector], distance: DistanceMetric): Int = {
    var bestIdx = 0
    var minD = Double.MaxValue
    var j = 0

    // while loop used intentionally over Scala collections to avoid boxing overhead
    while (j < prototypes.length) {
      val d = distance.compute(features, prototypes(j))
      if (d < minD) {
        minD = d
        bestIdx = j
      }
      j += 1
    }
    bestIdx
  }

  /** Array-based fast path for [[nearest]]. Uses `distanceUpTo` for early exit pruning.
   *  Ties are safely resolved to the first (lowest) index. */
  def nearestRaw(coords: Array[Double], prototypes: Array[Array[Double]], distance: DistanceMetric): Int = {
    var bestIdx = 0
    var minD = Double.MaxValue
    var j = 0

    while (j < prototypes.length) {
      val d = distance.distanceUpTo(coords, prototypes(j), minD)
      if (d < minD) {
        minD = d
        bestIdx = j
      }
      j += 1
    }
    bestIdx
  }
}
