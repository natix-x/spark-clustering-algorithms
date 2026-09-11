package clustering.algorithms.kmeans.hierarchical

import clustering.core.{Columns, Model}
import clustering.distance.DistanceMetric
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}

import scala.annotation.tailrec

/** A node of a bisecting k-means cluster tree. Every node carries the centroid of
 *  the whole subtree below it, which is what makes root-to-leaf traversal possible. */
sealed trait ClusterNode extends Serializable {
  def centroid: Vector
}

/** A leaf — one output cluster, labelled `clusterId`. */
final case class LeafNode(clusterId: Int, centroid: Vector) extends ClusterNode

/** A cluster that was split into two by a 2-means run. */
final case class InternalNode(centroid: Vector, left: ClusterNode, right: ClusterNode) extends ClusterNode

/** A fitted bisecting k-means model.
 *
 *  Labelling is **root-to-leaf traversal**, not nearest-leaf-centroid: at each
 *  internal node the point follows the closer child centroid. This matches MLlib's
 *  `BisectingKMeansModel` (kept as the validation reference) and costs O(depth)
 *  distance computations per point instead of O(k), but it can differ from a flat
 *  nearest-centroid assignment — a point may end up in a leaf whose centroid is not
 *  globally closest. That is a property of divisive hierarchical clustering, not a bug.
 *
 *  The tree is broadcast once per model and memoised across `assignClusters` calls,
 *  exactly like [[clustering.core.NearestPrototypeModel]] does for prototypes.
 */
class BisectingKMeansModel(
  val root:     ClusterNode,
  val distance: DistanceMetric
) extends Model {

  @transient private var bcRoot: Broadcast[ClusterNode] = _

  private def broadcast(data: DataFrame): Broadcast[ClusterNode] = synchronized {
    if (bcRoot == null) bcRoot = data.sparkSession.sparkContext.broadcast(root)
    bcRoot
  }

  override def assignClusters(data: DataFrame): DataFrame = {
    val bc         = broadcast(data)
    val dist       = distance
    // Coordinates unpacked ONCE per row (`toArray` is the vector's own array when dense), so the
    // O(depth) tree walk below runs the raw-array kernel at every level instead of re-dispatching
    // on the vector's type each time.
    val predictUDF = udf { features: Vector => BisectingKMeansModel.label(features.toArray, bc.value, dist) }
    data.withColumn(Columns.Prediction, predictUDF(col(Columns.Features)))
  }

  /** Leaf centroids in cluster-id order. */
  def clusterCentroids: Array[Vector] = {
    val leaves = BisectingKMeansModel.leaves(root)
    leaves.sortBy(_.clusterId).map(_.centroid).toArray
  }

  def numClusters: Int = BisectingKMeansModel.leaves(root).size
}

object BisectingKMeansModel {

  /** Cluster id of `coords`: descend to the closer child centroid at every level.
   *
   *  Uses `distanceUpToOrdinal` (unbounded, so no early exit — this compares two FULL distances,
   *  never a bound) purely to stay sqrt-free for Euclidean; only the argmin between two values
   *  matters here, never the value itself, same as [[clustering.core.NearestPrototypeModel.nearestRaw]]. */
  @tailrec
  def label(coords: Array[Double], node: ClusterNode, distance: DistanceMetric): Int = node match {
    case LeafNode(id, _) => id
    case InternalNode(_, left, right) =>
      // `<=` keeps ties going left, so labelling is deterministic.
      val leftD  = distance.distanceUpToOrdinal(coords, left.centroid.toArray, Double.MaxValue)
      val rightD = distance.distanceUpToOrdinal(coords, right.centroid.toArray, Double.MaxValue)
      val next = if (leftD <= rightD) left else right
      label(coords, next, distance)
  }

  def leaves(node: ClusterNode): Seq[LeafNode] = node match {
    case leaf: LeafNode               => Seq(leaf)
    case InternalNode(_, left, right) => leaves(left) ++ leaves(right)
  }
}
