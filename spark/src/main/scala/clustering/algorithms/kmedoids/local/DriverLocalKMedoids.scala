package clustering.algorithms.kmedoids.local

import clustering.algorithms.kmedoids.hybrid.{CLARA, PAMAE}
import clustering.algorithms.kmedoids.KMedoidsModel
import clustering.core.{Clusterer, Columns, Weights}
import clustering.distance.DistanceMetric
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.DataFrame

/** A k-medoids solver that keeps the O(n²) distance matrix in DRIVER memory ([[FastPAM]],
 *  [[FasterPAM]]).
 *
 *  `fitLocal` is the real entry point: [[CLARA]] and [[PAMAE]] run it on collected samples, which
 *  makes the exact solver a knob of the sampling methods (Schubert & Rousseeuw 2021 improve
 *  CLARA/CLARANS by replacing precisely it).
 */
private[kmedoids] trait DriverLocalKMedoids extends Clusterer {

  /** `weights(j)` = how many points `points(j)` stands for; the solver optimises
   *  Σ_j w_j · d(j, nearest medoid). */
  def fitLocal(points: Array[Vector], weights: Array[Double]): KMedoidsModel

  /** Unweighted convenience entry point. */
  final def fitLocal(points: Array[Vector]): KMedoidsModel =
    fitLocal(points, Weights.unit(points.length))

  /** Collects `[features, weight]` to the driver — only for data that fits in driver memory. */
  final override def fit(data: DataFrame): KMedoidsModel = {
    val rows = Weights.withWeights(data).collect()
    fitLocal(rows.map(_.getAs[Vector](Columns.Features)), rows.map(_.getDouble(1)))
  }
}

private[kmedoids] object DriverLocalKMedoids {

  /** Resolves an `inner` param value to a driver-local solver.
   *
   *  There is no `pam` entry: FastPAM1 searches identically to PAM and returns the identical
   *  medoids (Schubert & Rousseeuw 2021 present it as a pure O(k) runtime optimisation, not an
   *  approximation), so the exhaustive rung carried no information the fast one does not, at k²
   *  the cost. `fastpam` IS the exact baseline. */
  def fromName(name: String, k: Int, maxIter: Int, distance: DistanceMetric, seed: Long): DriverLocalKMedoids =
    name.toLowerCase match {
      case "fastpam" => new FastPAM(k, maxIter, distance)
      case "fasterpam" => new FasterPAM(k, maxIter, distance, seed)
      case other       => throw new IllegalArgumentException(
        s"Unknown inner k-medoids solver: '$other'. Known: fastpam, fasterpam")
    }
}
