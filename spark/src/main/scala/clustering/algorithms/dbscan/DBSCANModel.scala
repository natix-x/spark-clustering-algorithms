package clustering.algorithms.dbscan

import clustering.core.Model
import clustering.distance.DistanceMetric
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

import scala.collection.mutable.ArrayBuffer


/** A fitted DBSCAN model.
 *
 *  Holds the labelled CORE points only. Labelling of arbitrary points (the
 *  training set itself, border points, noise) is done by a grid join: a point
 *  is assigned the smallest cluster id among core points within `eps`, or `-1`
 *  (noise) if none. This avoids the O(n²) cartesian at prediction time.
 *
 *  @param labeledCorePoints columns `features` (Vector) and `clusterId` (Int)
 */
class DBSCANModel(
  val labeledCorePoints: DataFrame,
  val eps: Double,
  val distance: DistanceMetric
) extends Model {

  override def predict(features: Vector): Int =
    throw new UnsupportedOperationException(
      "DBSCANModel prefers labeledData (grid join). Single-point predict is not supported."
    )

  override def labeledData(data: DataFrame): DataFrame = {
    val e    = eps
    val dist = distance

    val cellUDF          = udf { (f: Vector) => DBSCANModel.cellKey(f, e) }
    val neighborCellsUDF = udf { (f: Vector) => DBSCANModel.neighborCellKeys(f, e) }
    val distUDF          = udf { (a: Vector, b: Vector) => dist.compute(a, b) }

    // Stable id so we can left-join predictions back and default noise to -1.
    val points = data.select(col("features"))
      .withColumn("rowId", monotonically_increasing_id())

    val newInCells = points.withColumn("cell", cellUDF(col("features")))

    val coreInCells = labeledCorePoints
      .withColumn("cell", explode(neighborCellsUDF(col("features"))))
      .select(col("cell"), col("features").as("coreFeatures"), col("clusterId"))

    val matched = newInCells
      .join(coreInCells, "cell")
      .filter(distUDF(col("features"), col("coreFeatures")) <= e)
      .groupBy("rowId")
      .agg(min("clusterId").as("matchedCluster"))

    points
      .join(matched, Seq("rowId"), "left_outer")
      .select(
        col("features"),
        coalesce(col("matchedCluster"), lit(-1)).as("prediction")
      )
  }
}

object DBSCANModel {

  /** Integer grid cell coordinates for a point at resolution `eps`. */
  private[dbscan] def cell(f: Vector, eps: Double): Array[Long] = {
    val v = f.toArray
    val c = new Array[Long](v.length)
    var i = 0
    while (i < v.length) {
      c(i) = math.floor(v(i) / eps).toLong
      i += 1
    }
    c
  }

  /** Stable string key for a point's own cell. */
  private[dbscan] def cellKey(f: Vector, eps: Double): String =
    cell(f, eps).mkString(",")

  /** Keys of all 3^d cells surrounding (and including) the point's own cell. */
  private[dbscan] def neighborCellKeys(f: Vector, eps: Double): Array[String] = {
    val base = cell(f, eps)
    val d    = base.length

    var combos: Array[Array[Long]] = Array(Array.empty[Long])
    var dim = 0
    while (dim < d) {
      val next = ArrayBuffer[Array[Long]]()
      var ci = 0
      while (ci < combos.length) {
        val prefix = combos(ci)
        var o = -1
        while (o <= 1) {
          next += (prefix :+ (base(dim) + o))
          o += 1
        }
        ci += 1
      }
      combos = next.toArray
      dim += 1
    }
    combos.map(_.mkString(","))
  }
}
