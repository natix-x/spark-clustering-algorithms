package clustering.algorithms.dbscan

import clustering.core.Clusterer
import clustering.distance.DistanceMetric
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.storage.StorageLevel
import org.graphframes.GraphFrame

/** Shared DBSCAN skeleton on top of DataFrames.
 *
 *  Concrete subclasses differ only in how they discover neighbour pairs
 *  (naive cartesian vs. grid-based spatial join). Everything downstream —
 *  core-point detection, connected components and cluster labelling — is
 *  common and expressed with DataFrame operations + GraphFrames.
 *
 *  Note: GraphFrames' default connected-components algorithm checkpoints
 *  intermediate results, so the caller must set a checkpoint dir
 *  (`sc.setCheckpointDir(...)`) before `fit`. See `SparkClusteringJob`.
 */
abstract class BaseDBSCAN(
  val eps: Double,
  val minPts: Int,
  val distance: DistanceMetric
) extends Clusterer {

  /** Returns neighbour id pairs as a DataFrame with columns `id1`, `id2`
   *  (id1 < id2) for every pair of points within `eps`.
   *  Input `indexed` carries columns `id` (Long) and `features` (Vector). */
  protected def findNeighborPairs(indexed: DataFrame): DataFrame

  override def fit(data: DataFrame): DBSCANModel = {
    val spark = data.sparkSession

    // Step 0: assign a stable unique id to each point.
    val indexed = data
      .select(col("features"))
      .withColumn("id", monotonically_increasing_id())
      .persist(StorageLevel.MEMORY_AND_DISK)

    // Delegate neighbour discovery to the concrete algorithm.
    val neighborPairs = findNeighborPairs(indexed).persist(StorageLevel.MEMORY_AND_DISK)

    // Step 1: core points — degree (+1 for the point itself) >= minPts.
    val degrees = neighborPairs.select(col("id1").as("id"))
      .union(neighborPairs.select(col("id2").as("id")))
      .groupBy("id").count()

    val corePoints = degrees
      .filter(col("count") + lit(1L) >= minPts)
      .select(col("id"))
      .persist(StorageLevel.MEMORY_AND_DISK)

    // Step 2: edges connecting two core points.
    val core1 = corePoints.select(col("id").as("c1"))
    val core2 = corePoints.select(col("id").as("c2"))
    val edges = neighborPairs
      .join(core1, col("id1") === col("c1"))
      .join(core2, col("id2") === col("c2"))
      .select(col("id1").as("src"), col("id2").as("dst"))

    // Step 3: connected components over the core graph (GraphFrames).
    val vertices   = corePoints.select(col("id")) // GraphFrame requires column "id"
    val cc         = GraphFrame(vertices, edges).connectedComponents.run()
    val coreLabels = cc.select(col("id"), col("component").as("label")) // label: Long

    // Step 4: densify arbitrary Long component ids into contiguous Int cluster ids.
    val labelMapping: Map[Long, Int] = coreLabels
      .select("label").distinct()
      .collect().map(_.getLong(0)).sorted.zipWithIndex.toMap
    val bcMap       = spark.sparkContext.broadcast(labelMapping)
    val toClusterId = udf { (l: Long) => bcMap.value(l) }

    // Step 5: labelled core points — the model re-predicts borders/noise via a
    // grid join at evaluation time (see DBSCANModel.labeledData).
    val labeledCorePoints = coreLabels
      .join(indexed, "id")
      .select(col("features"), toClusterId(col("label")).as("clusterId"))
      .persist(StorageLevel.MEMORY_AND_DISK)
    labeledCorePoints.count() // materialise before unpersisting upstream

    indexed.unpersist()
    neighborPairs.unpersist()
    corePoints.unpersist()

    new DBSCANModel(labeledCorePoints, eps, distance)
  }
}
