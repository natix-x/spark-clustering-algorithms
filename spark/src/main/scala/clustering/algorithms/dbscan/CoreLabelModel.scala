package clustering.algorithms.dbscan

import clustering.core.{Columns, Model}
import clustering.distance.DistanceMetric
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}


/** A fitted density model: labelled core points, plus the rule that labels everything else.
 *
 *  Assignment is one broadcast map — the core set is at most m points, so labelling needs no
 *  join and no second distributed pass.
 *
 *  @param corePoints        core-point coordinates
 *  @param coreClusterLabels cluster id of each core point; contiguous from 0, numbered by
 *                           ascending smallest candidate index, so independent of partitioning
 *  @param requireWithinEps  `true` (config `assign: eps`) = a point joins its closest core only
 *                           if that core is within ε, else `-1` — classic DBSCAN noise
 *                           semantics. `false` (`assign: closest`) = the paper's rule, which
 *                           assigns every point and so emits no noise at all.
 */
class CoreLabelModel(
  val corePoints: Array[Vector],
  val coreClusterLabels: Array[Int],
  val eps: Double,
  val distanceMetric: DistanceMetric,
  val requireWithinEps: Boolean
) extends Model {

  require(corePoints.length == coreClusterLabels.length,
    s"cores (${corePoints.length}) and labels (${coreClusterLabels.length}) must have the same length")

  /** Number of clusters found; 0 when the parameters produced no core point at all. */
  def numClusters: Int = if (coreClusterLabels.isEmpty) {
    0
  } else {
    coreClusterLabels.distinct.length
  }

  // Broadcast once, not per assignClusters call (evaluation makes several passes) and not
  // serialised into every task closure. Never destroyed: one run fits one model, so the
  // context stop reclaims it.
  @transient private var broadcastCorePoints:  Broadcast[Array[Vector]] = _
  @transient private var broadcastCoreLabels: Broadcast[Array[Int]]    = _

  override def assignClusters(data: DataFrame): DataFrame = {
    val sc = data.sparkSession.sparkContext
    synchronized {
      if (broadcastCorePoints == null)  broadcastCorePoints  = sc.broadcast(corePoints)
      if (broadcastCoreLabels == null) broadcastCoreLabels = sc.broadcast(coreClusterLabels)
    }
    val udfBroadcastCores = broadcastCorePoints
    val udfBroadcastLabels = broadcastCoreLabels
    val udfDistanceMetric = distanceMetric
    val udfEps = eps
    val udfRequireWithinEps = requireWithinEps

    val assignNearestClusterUDF = udf { features: Vector =>
      val availableCorePoints = udfBroadcastCores.value
      var nearestCoreIndex = -1
      var minDistanceToCore = Double.MaxValue
      var coreIndex = 0
      while (coreIndex < availableCorePoints.length) {
        val d = udfDistanceMetric.compute(features, availableCorePoints(coreIndex))
        if (d < minDistanceToCore) {
          minDistanceToCore = d
          nearestCoreIndex = coreIndex
        }
        coreIndex += 1
      }
      val isOutsideEpsilon = udfRequireWithinEps && (minDistanceToCore > udfEps)
      if (nearestCoreIndex < 0 || isOutsideEpsilon) {
        -1
      } else {
        udfBroadcastLabels.value(nearestCoreIndex)
      }
    }

    data.withColumn(Columns.Prediction, assignNearestClusterUDF(col(Columns.Features)))
  }
}
