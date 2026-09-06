package clustering.core

import org.apache.spark.sql.DataFrame

trait Clusterer extends Serializable {

  /** Fits on `data`, a frame with a `features` column (and optionally `weight`).
   *
   *  **The CALLER owns the cache.** Implementations must NOT persist the input `data`,
   *  as it is read multiple times and expected to be cached externally.
   *  An implementation may only cache intermediate frames it creates itself,
   *  and must unpersist them before returning.
   */
  def fit(data: DataFrame): Model
}
