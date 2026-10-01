package clustering.algorithms.coreset.inner

import clustering.algorithms.kmedoids.local.DriverLocalKMedoids
import clustering.distance.DistanceMetric
import org.apache.spark.ml.linalg.Vectors
import org.log4s.getLogger

/** `inner: medoids` — the exact k-medoids solver on the coreset (`innerSolver: fastpam |
 *  fasterpam`).
 *
 *  This is the entry that rescues the medoid objective at scale, and the third of the thesis'
 *  three principled reductions of it: uniform sampling (`clara`), one-batch estimation
 *  (`onebatchpam`), and importance-weighted reduction with an error bound (here). The solver is
 *  the same driver-local FastPAM1/FasterPAM that `clara` runs on its uniform samples, so the two
 *  entries differ in exactly one thing — HOW the m rows were chosen — which is what makes the
 *  comparison a controlled one.
 *
 *  Note the prototypes are real coreset rows, hence real data points, as the medoid objective
 *  requires.
 */
final class LocalMedoids(
  val solverName: String,
  val k: Int,
  val maxIter: Int,
  val distance: DistanceMetric,
  val seed: Long
) extends InnerSolver {

  private val logger = getLogger
  private val solver = DriverLocalKMedoids.fromName(solverName, k, maxIter, distance, seed)

  override val name: String = "medoids"
  override def fitDistance: DistanceMetric = distance
  override def modelDistance: DistanceMetric = distance

  override def solve(points: Array[Array[Double]], weights: Array[Double]): InnerSolver.Solution = {
    require(points.length >= k,
      s"coreset too small for the inner solve: got ${points.length} rows, need at least k=$k — raise m")
    val model = solver.fitLocal(points.map(Vectors.dense), weights)
    logger.info(s"coreset/medoids: k=$k solver=$solverName coresetRows=${points.length} " +
      s"iterationsRun=${model.iterationsRun}")
    InnerSolver.Solution(model.medoids.map(_.toArray), model.iterationsRun)
  }
}
