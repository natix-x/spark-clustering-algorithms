package clustering.core

import clustering.distance.{CosineDistance, DistanceMetric, EuclideanDistance, UnitSphereDistance}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}

/** The space a centroid-based algorithm optimises in — a knob, not an algorithm
 *  (euclidean = Lloyd k-means, spherical = Dhillon & Modha 2001). Metric is fixed
 *  by the geometry, not a free config param.
 *
 *  Three hooks:
 *    - [[prepare]]       — one transformation of the fit input (idempotent, so it is
 *                          safe for an outer algorithm such as bisecting k-means to
 *                          prepare once and let the inner k-means prepare again);
 *    - [[fitDistance]]   — the metric used inside the iteration loop, on prepared data;
 *    - [[modelDistance]] — the metric the fitted model uses to label RAW data, since
 *                          evaluation runs `assignClusters` on the untransformed
 *                          DataFrame (cosine is scale-invariant, so labels agree);
 *    - [[project]]       — the projection applied to each updated centroid.
 */
sealed trait Geometry extends Serializable {

  /** Registry name, as it appears in a run config. */
  def name: String

  def prepare(data: DataFrame): DataFrame

  def fitDistance: DistanceMetric

  def modelDistance: DistanceMetric

  def project(centroid: Vector): Vector
}

/** Plain Euclidean (Lloyd) geometry. */
object EuclideanGeometry extends Geometry {
  val name = "euclidean"

  override def prepare(data: DataFrame): DataFrame = data
  override def fitDistance: DistanceMetric = EuclideanDistance
  override def modelDistance: DistanceMetric = EuclideanDistance
  override def project(centroid: Vector): Vector = centroid
}

/** Spherical geometry (Dhillon & Modha 2001) — the unit hypersphere. */
object SphericalGeometry extends Geometry {
  val name = "spherical"

  override def prepare(data: DataFrame): DataFrame = {
    val normalise = udf { v: Vector => Geometry.l2Normalize(v) }
    data.withColumn(Columns.Features, normalise(col(Columns.Features)))
  }

  /** On normalised data cosine collapses to `1 − dot`. */
  override def fitDistance: DistanceMetric = UnitSphereDistance

  /** Cosine, not `1 − dot`: the model labels raw, un-normalised features and cosine
   *  is scale-invariant, so the labels are the same as on normalised data. */
  override def modelDistance: DistanceMetric = CosineDistance

  override def project(centroid: Vector): Vector = Geometry.l2Normalize(centroid)
}

object Geometry {

  /** L2-normalises `v`; a zero vector is returned unchanged (both cosine variants
   *  already treat it as maximally distant). */
  def l2Normalize(v: Vector): Vector = {
    val norm = Vectors.norm(v, 2.0)
    if (norm == 0.0) return v

    @inline def normalize(values: Array[Double]): Array[Double] = {
      val normalizedValues = values.clone()
      var i = 0
      while (i < normalizedValues.length) { normalizedValues(i) /= norm; i += 1 }
      normalizedValues
    }

    v match {
      case dv: DenseVector  =>
        Vectors.dense(normalize(dv.values))

      case sv: SparseVector =>
        Vectors.sparse(sv.size, sv.indices, normalize(sv.values))
    }
  }
}
