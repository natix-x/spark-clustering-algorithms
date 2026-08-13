package clustering.algorithms.kmeans.hierarchical

import clustering.algorithms.kmeans._
import clustering.algorithms.kmeans.hierarchical.BisectingKMeans.LeafState
import clustering.core._
import clustering.distance.DistanceMetric
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.storage.StorageLevel
import org.log4s.getLogger

import scala.collection.mutable.ArrayBuffer

/** Bisecting k-means (Steinbach, Karypis & Kumar 2000), divisive, built on [[KMeans]].
 *  Repeatedly splits the highest-cost leaf with a 2-means run until k leaves exist.
 */
class BisectingKMeans(
  val k: Int,
  val maxIter: Int = 20,
  val eps: Double = 1e-4,
  val seed: Long = 42L,
  val geometry: Geometry = EuclideanGeometry
) extends Clusterer {

  private val logger = getLogger

  override def fit(data: DataFrame): BisectingKMeansModel = {
    require(k >= 1, s"k must be >= 1, got $k")

    val fitDistance = geometry.fitDistance
    // Same rule as LloydKMeans: only a `prepare` that computes something earns its own cache;
    // otherwise this is a projection of the caller's cached data.
    val projectedRoot  = geometry.prepare(Weights.withWeights(data))
    val ownsRootCache  = geometry.transformsFeatures
    val preparedPoints = if (ownsRootCache) projectedRoot.persist(StorageLevel.MEMORY_AND_DISK) else projectedRoot

    val rootCentroid = geometry.project(calculateCentroid(preparedPoints))
    val rootNode = new BisectingKMeans.BuildingNode(rootCentroid)
    val (rootRows, rootMass, rootCost) = computeLeafStats(preparedPoints, rootCentroid, fitDistance)
    require(rootRows >= k, s"Dataset too small: n=$rootRows rows but k=$k clusters requested.")

    val leaves = ArrayBuffer(LeafState(rootNode, preparedPoints, rootRows, rootMass, rootCost, unsplittable = false))
    var splits = 0

    while (leaves.size < k) {
      // Highest-cost splittable leaf; `maxBy` keeps the first maximum, so ties resolve
      // by insertion order and the tree is reproducible.
      val candidates = leaves.indices.filter(i => canSplit(leaves(i)))
      if (candidates.isEmpty) {
        logger.warn(s"bisecting k-means: no splittable leaf left after ${leaves.size} clusters " +
          s"(requested k=$k) — stopping early")
        return buildModel(preparedPoints, ownsRootCache, leaves, rootNode)
      }
      val target = candidates.maxBy(i => leaves(i).cost)
      val leaf = leaves(target)

      // Split with the ordinary distributed KMeans; seed varies per split so the
      // bisections are independent yet reproducible.
      val bisector = new KMeans(k = 2, maxIter = maxIter, eps = eps,
        seed = seed + splits, geometry = geometry)
      val subModel = bisector.fit(leaf.points)

      // `assignedPoints` MUST be materialised before branching. Both childPartitions read it through their
      // own filter and each child's stats() is a separate action, so an uncached `assignedPoints`
      // makes Catalyst re-run the assignment UDF once per child — the 2-means assignment
      // computed twice for every point of the leaf. Worth reporting in the thesis: Spark's
      // lazy DAG re-executes a shared, uncached branch point, whereas a streaming pipeline
      // fans the same records out to both branches in a single pass.
      val assignedPoints = subModel.assignClusters(leaf.points).persist(StorageLevel.MEMORY_AND_DISK)

      val childCentroids = subModel.centroids
      val childPartitions = (0 until 2).map { branch =>
        val df = assignedPoints
          .filter(col(Columns.Prediction) === branch)
          .select(col(Columns.Features), col(Columns.Weight))
          .persist(StorageLevel.MEMORY_AND_DISK)
        val (rows, mass, c) = computeLeafStats(df, childCentroids(branch), fitDistance)
        (df, rows, mass, c)
      }

      // The childPartitions are materialised by stats() above, so the predictions are no longer needed.
      assignedPoints.unpersist(blocking = false)

      if (childPartitions.exists(_._2 == 0L)) {
        // Degenerate bisection — 2-means put everything in one child. Do not retry.
        childPartitions.foreach(_._1.unpersist(blocking = false))
        leaves(target) = leaf.copy(unsplittable = true)
        logger.warn(s"bisecting k-means: split of a ${leaf.pointCount}-row leaf (mass ${leaf.totalWeight}) " +
          s"produced an empty child; marking it unsplittable")
      } else {
        val leftNode  = new BisectingKMeans.BuildingNode(childCentroids(0))
        val rightNode = new BisectingKMeans.BuildingNode(childCentroids(1))
        leaf.node.left  = leftNode
        leaf.node.right = rightNode

        val (leftDf, leftRows, leftMass, leftCost)     = childPartitions.head
        val (rightDf, rightRows, rightMass, rightCost) = childPartitions(1)

        leaves(target) = LeafState(leftNode, leftDf, leftRows, leftMass, leftCost, unsplittable = false)
        leaves.insert(target + 1,
          LeafState(rightNode, rightDf, rightRows, rightMass, rightCost, unsplittable = false))

        // Children are materialised (the cost aggregation above forced them), so the
        // parent's cache is no longer needed.
        if (leaf.points ne preparedPoints) leaf.points.unpersist(blocking = false)
        splits += 1
      }
    }

    buildModel(preparedPoints, ownsRootCache, leaves, rootNode)
  }

  // A leaf can be split while it holds at least two rows and its points are not all identical.
  private def canSplit(leaf: LeafState): Boolean = !leaf.unsplittable && leaf.pointCount >= 2 && leaf.cost > 0.0

  /** Releases every cached subset and freezes the mutable tree into the model. */
  private def buildModel(
                      prepared: DataFrame,
                      ownsPreparedCache: Boolean,
                      leaves: ArrayBuffer[LeafState],
                      root: BisectingKMeans.BuildingNode
  ): BisectingKMeansModel = {
    leaves.foreach(l => if (l.points ne prepared) l.points.unpersist(blocking = false))
    if (ownsPreparedCache) prepared.unpersist(blocking = false)
    new BisectingKMeansModel(BisectingKMeans.freeze(root, Array(0)), geometry.modelDistance)
  }

  private def calculateCentroid(df: DataFrame): Vector =
    df.agg(Summarizer.mean(col(Columns.Features), col(Columns.Weight)).as("c"))
      .head().getAs[Vector]("c")

  /** Row count, weight mass Σ w, and WEIGHTED cost Σ w·d(x, centroid)² of `df` — all three in
   *  ONE Spark job.
   *
   *  The cost is the split criterion (weighted SSE under a Euclidean metric, its
   *  metric-generalised analogue otherwise). Rows and mass answer different questions: see
   *  [[BisectingKMeans.LeafState]]. */
  private def computeLeafStats(df: DataFrame, centroid: Vector, metric: DistanceMetric): (Long, Double, Double) = {
    val costUDF = udf { features: Vector =>
      val d = metric.compute(features, centroid)
      d * d
    }
    val row: Row = df
      .select(col(Columns.Weight),
              (costUDF(col(Columns.Features)) * col(Columns.Weight)).as("pointCost"))
      .agg(count(lit(1)).as("rows"),
           sum(Columns.Weight).as("mass"),
           sum("pointCost").as("cost"))
      .head()
    // sum over an empty set is null
    val mass = if (row.isNullAt(1)) 0.0 else row.getDouble(1)
    val cost = if (row.isNullAt(2)) 0.0 else row.getDouble(2)
    (row.getLong(0), mass, cost)
  }
}

private object BisectingKMeans {

  /** A leaf of the tree under construction: the mutable node, its point set, and the
   *  statistics that decide whether and when it gets split. Lives in the companion so
   *  it is a top-level case class (no outer reference to the fitting instance).
   *
   *  `rows` and `mass` are deliberately BOTH kept. Splittability is a question about rows —
   *  a 2-means run needs two distinct rows, and one row of weight 1000 cannot be bisected —
   *  while `mass` (Σ w) is what the leaf actually represents and what belongs in the logs.
   *  Weights may be fractional (lightweight coresets use w = 1/(m·q(x))), so `mass` is a
   *  Double and must never be used as a row count. */
  final case class LeafState(
                              node: BuildingNode,
                              points: DataFrame,
                              pointCount: Long,
                              totalWeight: Double,
                              cost: Double,
                              unsplittable: Boolean
  )

  /** Tree node under construction; leaves have both children null. Mutable because a
   *  split turns an existing leaf into an internal node in place. */
  final class BuildingNode(val centroid: Vector) {
    var left:  BuildingNode = _
    var right: BuildingNode = _
  }

  /** Freezes the mutable tree, numbering leaves in left-to-right DFS order so cluster
   *  ids depend only on the tree shape (determinism across runs). */
  private def freeze(node: BuildingNode, nextId: Array[Int]): ClusterNode =
    if (node.left == null) {
      val id = nextId(0)
      nextId(0) += 1
      LeafNode(id, node.centroid)
    } else {
      InternalNode(node.centroid, freeze(node.left, nextId), freeze(node.right, nextId))
    }
}
