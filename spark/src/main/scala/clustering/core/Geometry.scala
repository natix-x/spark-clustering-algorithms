package clustering.core

import clustering.distance.{CosineDistance, DistanceMetric, EuclideanDistance, UnitSphereDistance}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}

/** The space a centroid-based algorithm optimises in. Defines the distance metrics,
 *  projection, and objective functions (e.g., Euclidean vs Spherical).
 */
sealed trait Geometry extends Serializable {

  /** Registry name, as it appears in a run config. */
  def name: String
  def prepare(data: DataFrame): DataFrame
  /** True if `prepare` modifies features (as opposed to returning the input unchanged),
   *  signaling to the caller that the prepared frame might need its own cache. */
  def transformsFeatures: Boolean
  /** Metric used inside the iteration loop on prepared data. */
  def fitDistance: DistanceMetric
  /** Metric the fitted model uses to label RAW (unprepared) data. */
  def modelDistance: DistanceMetric
  /** Projection applied to each updated centroid. */
  def project(centroid: Vector): Vector
  /** One point's contribution to the objective THIS geometry's iteration optimises.
   *  - Euclidean minimises SSE, so this returns the squared distance (d²).
   *  - Spherical maximises cosine similarity, so this returns the first power (d). */
  def pointCost(distance: Double): Double
  /** Inverse of [[pointCost]], converting a mean cost back into a typical distance. */
  def costToDistance(meanCost: Double): Double
}

/** Plain Euclidean (Lloyd) geometry. */
object EuclideanGeometry extends Geometry {
  val name: String = "euclidean"

  override def prepare(data: DataFrame): DataFrame = data
  override def transformsFeatures: Boolean = false
  override def fitDistance: DistanceMetric = EuclideanDistance
  override def modelDistance: DistanceMetric = EuclideanDistance
  override def project(centroid: Vector): Vector = centroid
  override def pointCost(distance: Double): Double = distance * distance
  override def costToDistance(meanCost: Double): Double = math.sqrt(meanCost)
}

/** Spherical geometry (unit hypersphere). */
object SphericalGeometry extends Geometry {
  val name: String = "spherical"

  private val normalizeUdf = udf(Geometry.l2Normalize _)

  override def prepare(data: DataFrame): DataFrame =
    data.withColumn(Columns.Features, normalizeUdf(col(Columns.Features)))
  override def transformsFeatures: Boolean = true
  /** On normalised data, cosine collapses to `1 − dot`. */
  override def fitDistance: DistanceMetric = UnitSphereDistance
  /** Cosine is scale-invariant, so it safely labels raw, un-normalised data. */
  override def modelDistance: DistanceMetric = CosineDistance
  override def project(centroid: Vector): Vector = Geometry.l2Normalize(centroid)
  override def pointCost(distance: Double): Double = distance
  override def costToDistance(meanCost: Double): Double = meanCost
}

object Geometry {

  /** L2-normalises `v`. A zero vector is returned unchanged. */
  def l2Normalize(v: Vector): Vector = {
    val norm = Vectors.norm(v, 2.0)
    if (norm == 0.0) return v

    @inline def normalize(values: Array[Double]): Array[Double] = {
      val out = values.clone()
      var i = 0
      // while loop used intentionally over .map() to avoid boxing overhead in critical ML paths
      while (i < out.length) {
        out(i) /= norm
        i += 1
      }
      out
    }

    v match {
      case dv: DenseVector  => Vectors.dense(normalize(dv.values))
      case sv: SparseVector => Vectors.sparse(sv.size, sv.indices, normalize(sv.values))
    }
  }
}
