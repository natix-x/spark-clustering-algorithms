package clustering.core

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Column, DataFrame}
import org.apache.spark.sql.functions.{coalesce, col, isnan, lit, sum, when}
import org.apache.spark.sql.types.DoubleType

/** Utilities for handling point weights. Unweighted data defaults to a weight of 1.0. */
object Weights {

  def isWeighted(data: DataFrame): Boolean = data.columns.contains(Columns.Weight)

  /** Returns the weight column (cast to Double) or a literal 1.0 if absent. */
  def column(data: DataFrame): Column =
    if (isWeighted(data)) col(Columns.Weight).cast(DoubleType) else lit(1.0)

  /** Sanitises weights by replacing NULL, NaN, negative, and infinite values with 0.0.
   *  This prevents corrupted records from poisoning cluster aggregates (e.g., means, counts).
   *  Folds to `lit(1.0)` on unweighted input. */
  def safeColumn(data: DataFrame): Column = {
    val w = coalesce(column(data), lit(0.0))
    when(!isnan(w) && w > lit(0.0) && w < lit(Double.PositiveInfinity), w)
      .otherwise(lit(0.0))
  }

  /** Projects the DataFrame to exactly `[features, weight]`.
   *  This is a bare projection; callers should avoid caching the result unnecessarily. */
  def withWeights(data: DataFrame): DataFrame =
    data.select(col(Columns.Features), column(data).as(Columns.Weight))

  /** RDD counterpart of [[withWeights]]. */
  def toRdd(data: DataFrame): RDD[(Vector, Double)] =
    withWeights(data).rdd.map(r => (r.getAs[Vector](0), r.getDouble(1)))

  /** Creates an array of 1.0s of size `n`. */
  def unit(n: Int): Array[Double] = Array.fill(n)(1.0)

  /** Computes the sum of all weights (the total mass the frame represents).
   *  Requires a full aggregation pass. */
  def totalMass(data: DataFrame): Double = {
    val row = data.agg(sum(column(data)).as("mass")).head()
    if (row.isNullAt(0)) 0.0 else row.getDouble(0)
  }
}
