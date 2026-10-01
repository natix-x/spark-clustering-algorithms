package clustering.algorithms.coreset

import clustering.core.NearestPrototypeModel
import clustering.distance.DistanceMetric
import org.apache.spark.ml.linalg.Vector
import org.log4s.getLogger

/** A model fitted on a coreset: k prototypes, whatever the inner algorithm was.
 *
 *  `inner: kmeans` gives centroids, `inner: medoids` real data points, `inner: hac` the weighted
 *  centres of the agglomerative clusters — all three label the full data by nearest prototype, so
 *  the coreset itself never has to be carried into the labelling pass (see [[inner.Agglomerative]]
 *  for why that matters).
 *
 *  @param requestedSize the `m` the run asked for, kept beside [[coresetSize]] because the draw is
 *                       Poisson and its achieved size is only `m` in expectation — quoting a
 *                       quality-vs-m curve against the requested value when the realised one
 *                       differs is exactly the mistake the silhouette's own counters exist to
 *                       prevent
 */
class CoresetModel(
  prototypesArg: Array[Vector],
  distanceArg: DistanceMetric,
  iterationsRunArg: Option[Int],
  val coresetSize: Int,
  val requestedSize: Int,
  val inner: String
) extends NearestPrototypeModel(prototypesArg, distanceArg) {

  private val logger = getLogger

  logger.debug(s"CoresetModel created k=${prototypesArg.length} inner=$inner " +
    s"coresetSize=$coresetSize requestedSize=$requestedSize iterationsRun=$iterationsRunArg")

  /** The cluster prototypes, in cluster-id order. */
  def prototypeVectors: Array[Vector] = prototypes

  override val iterationsRun: Option[Int] = iterationsRunArg

  /** The realised draw, not `requestedSize`: the Poisson draw makes the two differ by a random
   *  amount, and the quality a run reports belongs to the rows it actually clustered. */
  override val reductionSize: Option[Int] = Some(coresetSize)
}
