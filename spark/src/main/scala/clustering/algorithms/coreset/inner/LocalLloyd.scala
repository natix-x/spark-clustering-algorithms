package clustering.algorithms.coreset.inner

import clustering.core.{Geometry, NearestPrototypeModel}
import clustering.distance.DistanceMetric
import org.apache.spark.ml.linalg.{Vectors}
import org.apache.spark.sql.DataFrame
import org.log4s.getLogger

import scala.util.Random

/** `inner: kmeans` — Lloyd's iteration on the coreset, driver-local.
 *
 *  Same loop as the distributed [[clustering.algorithms.kmeans.LloydKMeans]] (weighted mean,
 *  geometry projection, `eps` on the maximum centroid movement, `maxIter` as a CAP), with the
 *  per-row fold running over the coreset's arrays instead of an RDD. A coreset of a few thousand
 *  rows makes the distributed loop's per-iteration job deployment the dominant cost, which is the
 *  whole point of the reduction — so the inner solve deliberately does NOT reuse the distributed
 *  entry.
 *
 *  The geometry is the same knob the `kmeans` entry has, and it applies to the WHOLE pipeline: the
 *  data is projected before the coreset is drawn, so the importance sampling measures distances in
 *  the space the inner algorithm optimises in.
 */
final class LocalLloyd(
  val k: Int,
  val maxIter: Int,
  val eps: Double,
  val geometry: Geometry,
  val seed: Long
) extends InnerSolver {

  private val logger = getLogger

  override val name: String = "kmeans"
  override def fitDistance: DistanceMetric = geometry.fitDistance
  override def modelDistance: DistanceMetric = geometry.modelDistance
  override def prepare(data: DataFrame): DataFrame = geometry.prepare(data)
  override def transformsFeatures: Boolean = geometry.transformsFeatures

  override def solve(points: Array[Array[Double]], weights: Array[Double]): InnerSolver.Solution = {
    require(points.length >= k,
      s"coreset too small for the inner solve: got ${points.length} rows, need at least k=$k — raise m")
    iterate(points, weights, LocalLloyd.initialCentroids(points, weights, k, seed))
  }

  /** The loop itself, from an EXPLICIT init.
   *
   *  Separate from [[solve]] because the init is a weighted random draw, and a weighted input and
   *  its duplicated equivalent necessarily draw different starting points — so the
   *  "weighting == duplication" invariant can only be checked with the init held fixed. Same
   *  reasoning (and the same seam) as the medoid refinement primitive's in `WeightedMedoidsSpec`. */
  private[coreset] def iterate(
    points: Array[Array[Double]],
    weights: Array[Double],
    initialCentroids: Array[Array[Double]]
  ): InnerSolver.Solution = {
    val dim = points.head.length
    val distance = geometry.fitDistance

    var centroids = initialCentroids
    var iteration = 0
    var converged = false

    while (!converged && iteration < maxIter) {
      val sums = Array.fill(k)(new Array[Double](dim))
      val masses = new Array[Double](k)

      var j = 0
      while (j < points.length) {
        val row = points(j)
        val w = weights(j)
        val cluster = NearestPrototypeModel.nearestRaw(row, centroids, distance)
        val acc = sums(cluster)
        var i = 0
        while (i < dim) { acc(i) += w * row(i); i += 1 }
        masses(cluster) += w
        j += 1
      }

      val updated = Array.tabulate(k) { cluster =>
        val mass = masses(cluster)
        if (mass <= 0.0) centroids(cluster)  // empty cluster keeps its centre
        else {
          val mean = Array.tabulate(dim)(i => sums(cluster)(i) / mass)
          geometry.project(Vectors.dense(mean)).toArray
        }
      }

      var maxMovement = 0.0
      var cluster = 0
      while (cluster < k) {
        val movement = distance.compute(centroids(cluster), updated(cluster))
        if (movement > maxMovement) maxMovement = movement
        cluster += 1
      }

      centroids = updated
      iteration += 1
      converged = maxMovement < eps
      logger.debug(f"coreset/kmeans: iteration=$iteration maxMovement=$maxMovement%.6f converged=$converged")
    }

    logger.info(s"coreset/kmeans: k=$k iterationsRun=$iteration converged=$converged " +
      s"coresetRows=${points.length} geometry=${geometry.name}")
    InnerSolver.Solution(centroids, Some(iteration))
  }
}

object LocalLloyd {

  /** A-Res weighted reservoir sampling — `score = rand^(1/w)`, keep the k highest — the same rule
   *  [[clustering.algorithms.kmeans.LloydKMeans.sampleInitialCentroids]] uses on the distributed
   *  side, so a coreset row of weight w seeds exactly as w duplicated rows would. Unit weights
   *  collapse it to a plain uniform draw.
   *
   *  Deterministic given the seed AND the row order, which is per-engine (Spark collects in
   *  partition order, Flink assembles in subtask order): sample IDENTITY was never a cross-engine
   *  invariant, only its distribution. */
  def initialCentroids(
    points: Array[Array[Double]],
    weights: Array[Double],
    k: Int,
    seed: Long
  ): Array[Array[Double]] = {
    val rng = new Random(seed)
    val scored = Array.tabulate(points.length) { j =>
      val w = math.max(weights(j), Double.MinPositiveValue)
      (math.pow(rng.nextDouble(), 1.0 / w), j)
    }
    scored
      .sortBy { case (score, index) => (-score, index) }
      .take(k)
      .map { case (_, index) => points(index).clone() }
  }
}
