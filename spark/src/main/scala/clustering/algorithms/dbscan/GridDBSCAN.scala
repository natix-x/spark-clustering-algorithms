package clustering.algorithms.dbscan

import clustering.distance.{DistanceMetric, EuclideanDistance}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

/** Grid-based DBSCAN — points are bucketed into ε-sized spatial cells and
 *  only compared against the 3^d neighbouring cells, turning the O(n²)
 *  cartesian into a much cheaper equi-join on the cell key (which Catalyst
 *  can plan/shuffle efficiently). */
class GridDBSCAN(
  eps: Double,
  minPts: Int,
  distance: DistanceMetric = EuclideanDistance
) extends BaseDBSCAN(eps, minPts, distance) {

  override protected def findNeighborPairs(indexed: DataFrame): DataFrame = {
    val e       = eps
    val dist    = distance
    val distUDF          = udf { (a: Vector, b: Vector) => dist.compute(a, b) }
    val cellUDF          = udf { (f: Vector) => DBSCANModel.cellKey(f, e) }
    val neighborCellsUDF = udf { (f: Vector) => DBSCANModel.neighborCellKeys(f, e) }

    // Left side: each point fanned out across its own + neighbouring cells.
    val left = indexed
      .withColumn("cell", explode(neighborCellsUDF(col("features"))))
      .select(col("id").as("id1"), col("features").as("f1"), col("cell"))

    // Right side: each point keyed by its own cell.
    val right = indexed
      .withColumn("cell", cellUDF(col("features")))
      .select(col("id").as("id2"), col("features").as("f2"), col("cell"))

    left.join(right, "cell")
      .filter(col("id1") < col("id2"))          // dedup + drop self-pairs
      .filter(distUDF(col("f1"), col("f2")) <= e)
      .select(col("id1"), col("id2"))
  }
}
