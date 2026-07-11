package clustering.core

import org.apache.spark.sql.DataFrame


/** A fitted clustering model.
 *
 *  The sole contract is [[assignClusters]]: given a DataFrame with a
 *  [[Columns.Features]] column, return it with a [[Columns.Prediction]] column
 *  holding each row's cluster id.
 *
 *  Single-point prediction is intentionally NOT part of the contract — some
 *  models (e.g. DBSCAN) only support batch labelling via a distributed join and
 *  have no meaningful per-point predict.
 */
trait Model extends Serializable {
  def assignClusters(data: DataFrame): DataFrame
}
