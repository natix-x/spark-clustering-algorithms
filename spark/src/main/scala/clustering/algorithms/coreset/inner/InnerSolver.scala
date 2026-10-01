package clustering.algorithms.coreset.inner

import clustering.distance.DistanceMetric
import org.apache.spark.sql.DataFrame

/** The clustering that runs ON the coreset, driver-local.
 *
 *  This is the `inner` knob of the `coreset` registry entry, and the reason the slot is the
 *  highest-leverage one on the target list: the reduction is written once and every inner
 *  algorithm composes with it. `m` points are already in driver memory once the coreset is drawn,
 *  so an inner solve is a local computation on raw arrays in BOTH engines — an O(m^2) linkage or an
 *  O(m k iter) Lloyd on a few thousand rows is microseconds-per-round work that a distributed job's
 *  fixed cost (1.3 s on Spark, 4.9 s on Flink, measured on Ares) would dwarf many times over.
 *
 *  Two implementations, mirrored in the Flink engine file for file: [[LocalLloyd]]
 *  (`inner: kmeans`) and [[LocalMedoids]] (`inner: medoids`).
 *
 *  A third, `hac` (weighted agglomerative clustering by nearest-neighbour chain, cut at k), was
 *  written for both engines and then CUT on 14.09.2026 with its reason stated — see
 *  `algorithm_selection.md` §8. In short: an agglomerative inner is entirely local, so its cost
 *  curve is flat in worker count and it measures nothing on the thesis' performance axis that
 *  `inner: kmeans` does not already measure through the shared construction, while the
 *  hierarchical paradigm is already represented by the divisive `bisectingkmeans`. Recoverable
 *  from that note if a later pass wants the agglomerative rung back.
 */
trait InnerSolver extends Serializable {

  /** The `inner` param value that selects this solver. */
  def name: String

  /** Metric the COREST CONSTRUCTION and the inner solve run under, on prepared data. */
  def fitDistance: DistanceMetric

  /** Metric the fitted model labels RAW (unprepared) data with. Differs from [[fitDistance]]
   *  only where the geometry transforms features (spherical: unit-sphere in, cosine out). */
  def modelDistance: DistanceMetric

  /** Projection applied to the whole dataset before the coreset is drawn. Identity unless the
   *  inner algorithm optimises in a transformed space. */
  def prepare(data: DataFrame): DataFrame = data

  /** True when [[prepare]] computes something, so the caller knows the prepared frame needs its
   *  own cache (the coreset construction reads it three times). */
  def transformsFeatures: Boolean = false

  /** Clusters the weighted coreset. `weights(j)` is the mass `points(j)` stands for, so the
   *  objective optimised is `sum_j w_j * d(j, nearest prototype)` — the "weighting == duplication"
   *  invariant the whole engine is built on. */
  def solve(points: Array[Array[Double]], weights: Array[Double]): InnerSolver.Solution
}

object InnerSolver {

  /** @param prototypes    cluster centres in cluster-id order; k of them, so the fitted model is a
   *                       plain nearest-prototype model whatever the inner algorithm was
   *  @param iterationsRun the inner convergence loop's round count, empty for a solver that has
   *                       none */
  final case class Solution(prototypes: Array[Array[Double]], iterationsRun: Option[Int])

}
