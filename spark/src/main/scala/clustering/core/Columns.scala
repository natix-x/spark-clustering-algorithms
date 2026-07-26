package clustering.core

/** Canonical column names shared across the clustering pipeline.
 *
 *  Every data source produces a `features` column and every model produces a
 *  `prediction` column; centralising the names here keeps them from drifting
 *  apart as string literals scattered through the codebase.
 */
object Columns {
  val Features   = "features"
  val Prediction = "prediction"

  /** Optional point weight; absent means every point weighs 1.0. See [[Weights]]. */
  val Weight     = "weight"
}