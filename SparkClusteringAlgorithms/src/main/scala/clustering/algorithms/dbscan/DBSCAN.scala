package clustering.algorithms.dbscan

import clustering.distance.{DistanceMetric, EuclideanDistance}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

/** Naive DBSCAN — neighbour discovery via a full cross join (O(n²)). */
class DBSCAN(
  eps: Double,
  minPts: Int,
  distance: DistanceMetric = EuclideanDistance
) extends BaseDBSCAN(eps, minPts, distance) {

  override protected def findNeighborPairs(indexed: DataFrame): DataFrame = {
    val e       = eps
    val dist    = distance
    val distUDF = udf { (a: Vector, b: Vector) => dist.compute(a, b) }

    val left  = indexed.select(col("id").as("id1"), col("features").as("f1"))
    val right = indexed.select(col("id").as("id2"), col("features").as("f2"))

    left.crossJoin(right)
      .filter(col("id1") < col("id2"))          // dedup + drop self-pairs
      .filter(distUDF(col("f1"), col("f2")) <= e)
      .select(col("id1"), col("id2"))
  }
}
