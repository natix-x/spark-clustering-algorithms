package clustering.core

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Column, DataFrame}
import org.apache.spark.sql.functions.{col, lit}
import org.apache.spark.sql.types.DoubleType

/** Point weights: optional, invisible when absent (defaults to 1.0). */
object Weights {

  def isWeighted(data: DataFrame): Boolean = data.columns.contains(Columns.Weight)

  def column(data: DataFrame): Column =
    if (isWeighted(data)) col(Columns.Weight).cast(DoubleType) else lit(1.0)

  /** Projects to exactly `[features, weight]`. */
  def withWeights(data: DataFrame): DataFrame =
    data.select(col(Columns.Features), column(data).as(Columns.Weight))

  /** RDD-side counterpart of [[withWeights]], for
   *  [[clustering.algorithms.kmedoids.DistributedFastPAM]] only. */
  def toRdd(data: DataFrame): RDD[(Vector, Double)] =
    withWeights(data).rdd.map(r => (r.getAs[Vector](0), r.getDouble(1)))

  def unit(n: Int): Array[Double] = Array.fill(n)(1.0)
}
