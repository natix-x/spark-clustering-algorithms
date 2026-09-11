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
 *  Repeatedly splits one leaf with a 2-means run until k leaves exist.
 *
 *  `select` is which leaf that is, and the literature offers exactly two answers — both
 *  reachable here, because they optimise different things rather than one being better:
 *  `cost` splits the leaf of highest weighted SSE (minimises error, and is scikit-learn's
 *  default `bisecting_strategy = "biggest_inertia"`), `size` splits the largest one (tends
 *  towards a balanced tree, and is what Steinbach et al. actually ran — "we found little
 *  difference between the possible methods for selecting a cluster to split and chose to split
 *  the largest remaining cluster"). `cost` is the default because it is the criterion the tree
 *  is otherwise built and scored against.
 *
 *  BOTH are weighted: `size` reads mass Σw, not the row count, because this codebase's standing
 *  invariant is that weighting equals duplication — a row of weight w represents w rows, and a
 *  criterion that counted it once would break that. Neither source defines the weighted case at
 *  all (they predate it), and on unweighted data mass equals rows, so the paper's criterion is
 *  reproduced exactly where the paper applies. Row count is still kept in `LeafState`, because
 *  SPLITTABILITY is genuinely a question about rows: 2-means needs two distinct rows, and one
 *  row of weight 1000 cannot be bisected however much mass it carries.
 *
 *  `trials` is the paper's ITER: how many independent 2-means runs compete for each split,
 *  the cheapest by total child cost winning. The default 1 is the naive rung — one shot per
 *  split, no retry — and higher values buy back the sensitivity to initialisation that a
 *  single shot leaves in. The knob exists because that sensitivity is a MEASUREMENT here:
 *  the leaf choice is `maxBy(cost)` over floating-point sums, so a perturbation that would
 *  only move boundary points in flat k-means can reorder two near-equal-cost leaves and
 *  change the shape of the whole tree. Sweeping `trials` turns "the labels moved" into a
 *  cost-vs-stability curve.
 *
 *  It does NOT buy a monotonically better clustering, and the sweep must not be read as if it
 *  did. Each trial optimises one bisection, but the tree is greedy: the leaf split next is
 *  whichever costs most at that moment, so a better first cut changes every later choice and
 *  can land on a worse final tree. Pinned by `BisectingTrialsSpec`, which asserts the local
 *  guarantee at k = 2 and the global anomaly at k = 6.
 */
class BisectingKMeans(
  val k: Int,
  val maxIter: Int = 20,
  val eps: Double = 1e-4,
  val seed: Long = 42L,
  val geometry: Geometry = EuclideanGeometry,
  val trials: Int = 1,
  val select: String = "cost"
) extends Clusterer {

  private val logger = getLogger

  /** Resolved once, at construction, so a bad `select` fails before any Spark job is submitted
   *  rather than after the root statistics have been paid for. */
  private val leafScore: LeafState => Double = select.toLowerCase match {
    case "cost" => _.cost
    // Mass Σw, not the row count: the repo-wide invariant is that weighting IS duplication, so
    // a row of weight w stands for w rows and "largest cluster" has to count it as such.
    // On unweighted data the two coincide, so this reproduces the paper exactly there.
    case "size" => _.totalWeight
    case other  => throw new IllegalArgumentException(
      s"Unknown leaf selection: '$other'. Known: cost, size")
  }

  override def fit(data: DataFrame): BisectingKMeansModel = {
    require(k >= 1, s"k must be >= 1, got $k")
    require(trials >= 1, s"trials must be >= 1, got $trials")

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
      // Best splittable leaf under `select`; `maxBy` keeps the first maximum, so ties resolve
      // by insertion order and the tree is reproducible. Ties are the common case under
      // `size` (equal row counts) and vanishingly rare under `cost`.
      val candidates = leaves.indices.filter(i => canSplit(leaves(i)))
      if (candidates.isEmpty) {
        logger.warn(s"bisecting k-means: no splittable leaf left after ${leaves.size} clusters " +
          s"(requested k=$k) — stopping early")
        return buildModel(preparedPoints, ownsRootCache, leaves, rootNode)
      }
      val target = candidates.maxBy(i => leafScore(leaves(i)))
      val leaf = leaves(target)

      val subModel = bestBisection(leaf.points, splits, fitDistance)

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

  /** The paper's ITER step: run `trials` independent 2-means bisections of `points` and keep
   *  the one whose two children cost least in total.
   *
   *  Each trial gets its own seed, derived from the split index so the whole tree stays
   *  reproducible. At `trials = 1` the seed reduces to `seed + splitIndex` and the scoring job
   *  is skipped entirely — the default path is exactly the single-shot bisection, not a
   *  one-element search that pays for a comparison it cannot lose.
   *
   *  Losing trials are scored but never materialised: [[splitCost]] needs only the two
   *  centroids, so a trial costs one 2-means fit plus one aggregation, not a pair of cached
   *  child subsets. */
  private def bestBisection(points: DataFrame, splitIndex: Int, metric: DistanceMetric): KMeansModel = {
    def bisect(trial: Int): KMeansModel =
      new KMeans(k = 2, maxIter = maxIter, eps = eps,
        seed = seed + splitIndex.toLong * trials + trial, geometry = geometry).fit(points)

    if (trials == 1) bisect(0)
    else {
      var best = bisect(0)
      var bestCost = splitCost(points, best.centroids, metric)
      var trial = 1
      while (trial < trials) {
        val candidate = bisect(trial)
        val candidateCost = splitCost(points, candidate.centroids, metric)
        // Strict `<` keeps the earliest of several equally good trials, so ties do not
        // depend on which one happened to be evaluated last.
        if (candidateCost < bestCost) { best = candidate; bestCost = candidateCost }
        trial += 1
      }
      best
    }
  }

  /** Total weighted cost of a candidate bisection: Σ w·φ(min_j d(x, c_j)), i.e. the sum of what
   *  both children would report from [[computeLeafStats]] — computed in ONE job, without
   *  materialising either child. */
  private def splitCost(df: DataFrame, centroids: Array[Vector], metric: DistanceMetric): Double = {
    // Unpacked once on the driver, like every other scan in the codebase, so the UDF runs the
    // raw-array kernel with its shrinking-bound early exit.
    val raw = centroids.map(_.toArray)
    val geom = geometry // local val: the UDF must not capture the enclosing clusterer
    val costUDF = udf { features: Vector =>
      val coords = features.toArray
      var min = Double.MaxValue
      var j = 0
      while (j < raw.length) {
        val d = metric.distanceUpToOrdinal(coords, raw(j), min)
        if (d < min) min = d
        j += 1
      }
      geom.pointCostFromOrdinal(min)
    }
    val row: Row = df
      .select((costUDF(col(Columns.Features)) * col(Columns.Weight)).as("pointCost"))
      .agg(sum("pointCost").as("cost"))
      .head()
    if (row.isNullAt(0)) 0.0 else row.getDouble(0)
  }

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

  /** Row count, weight mass Σ w, and WEIGHTED cost Σ w·φ(d(x, centroid)) of `df` — all three in
   *  ONE Spark job.
   *
   *  The cost is the split criterion, and φ comes from the geometry rather than being a fixed
   *  square, so it is always the objective the inner k-means actually optimises — weighted SSE
   *  under euclidean, Σ w·(1 − cos) under spherical (see [[Geometry.pointCost]]). Rows and mass
   *  answer different questions: see [[BisectingKMeans.LeafState]]. */
  private def computeLeafStats(df: DataFrame, centroid: Vector, metric: DistanceMetric): (Long, Double, Double) = {
    val geom = geometry // local val: the UDF must not capture the enclosing clusterer
    // Raw-array + ordinal, like every other scan in the codebase: no Vector-overload dispatch,
    // no sqrt paid just to be squared back by `pointCost` for Euclidean.
    val centroidArr = centroid.toArray
    val costUDF = udf { features: Vector =>
      val ordinal = metric.distanceUpToOrdinal(features.toArray, centroidArr, Double.MaxValue)
      geom.pointCostFromOrdinal(ordinal)
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
