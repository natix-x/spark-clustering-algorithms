package clustering.algorithms.dbscan

import clustering.core.{Columns, Model}
import clustering.distance.DistanceMetric
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}


/** A fitted density model: labelled core points, plus the rule that labels everything else — step
 *  4 of DBSCAN++. Assignment semantics and rationale: `docs/dbscanpp_docs.md` §5.
 *
 *  @param corePoints core-point coordinates
 *  @param coreClusterLabels cluster id of each core point; contiguous from 0, numbered by
 *                           ascending smallest candidate index, so independent of partitioning
 *  @param requireWithinEps `true` (config `assign: eps`) = classic DBSCAN noise semantics.
 *                           `false` (`assign: closest`) = the paper's rule, no noise at all.
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

  def getNumberOfClusters: Int = if (coreClusterLabels.isEmpty) {
    0
  } else {
    coreClusterLabels.distinct.length
  }

  // Lazily broadcast once and cached (assignClusters may run several times); never destroyed —
  // one run fits one model, so the SparkContext shutdown reclaims it.
  @transient private var broadcastCorePointsCoords: Broadcast[Array[Array[Double]]] = _
  @transient private var broadcastCorePointsLabels: Broadcast[Array[Int]]           = _

  override def assignClusters(data: DataFrame): DataFrame = {
    val sc = data.sparkSession.sparkContext
    synchronized {
      if (broadcastCorePointsCoords == null) broadcastCorePointsCoords = sc.broadcast(corePoints.map(_.toArray))
      if (broadcastCorePointsLabels == null) broadcastCorePointsLabels = sc.broadcast(coreClusterLabels)
    }
    val udfBroadcastCores = broadcastCorePointsCoords
    val udfBroadcastLabels = broadcastCorePointsLabels
    val udfDistanceMetric = distanceMetric
    val udfEps = eps
    val udfRequireWithinEps = requireWithinEps

    val assignNearestClusterUDF = udf { features: Vector =>
      val availableCorePoints = udfBroadcastCores.value
      val pointCoords = features.toArray
      var nearestCoreIndex = -1
      // Starts at ε when requireWithinEps: a nearest core beyond ε is noise regardless, so it
      // need not be measured exactly. See docs/dbscanpp_docs.md §5.
      var searchBound = if (udfRequireWithinEps) udfEps else Double.PositiveInfinity
      var coreIndex = 0
      while (coreIndex < availableCorePoints.length) {
        val d = udfDistanceMetric.distanceUpTo(pointCoords, availableCorePoints(coreIndex), searchBound)
        // `<` keeps the FIRST of several equidistant cores — ties resolve by ascending core index.
        if (d < searchBound || (nearestCoreIndex < 0 && d <= searchBound)) {
          searchBound = d
          nearestCoreIndex = coreIndex
        }
        coreIndex += 1
      }
      if (nearestCoreIndex < 0) -1 else udfBroadcastLabels.value(nearestCoreIndex)
    }

    data.withColumn(Columns.Prediction, assignNearestClusterUDF(col(Columns.Features)))
  }
}
