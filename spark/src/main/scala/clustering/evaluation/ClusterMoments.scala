package clustering.evaluation

import clustering.core.{Columns, Model, Weights}
import clustering.distance.DistanceMetric
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType

/** Per-cluster first and second moments about the cluster's own centroid — shared by
 *  [[DaviesBouldinIndex]] and [[CalinskiHarabaszIndex]] so the two passes run once for the pair,
 *  not once per index.
 *
 *  O(n·d), two linear passes — computed on the FULL dataset, unlike the sampled silhouette.
 *
 *  @param clusterMoments one entry per non-noise cluster, ascending by label
 *  @param globalCentroid weighted mean of the non-noise points (the grand centroid CH needs)
 */
final case class ClusterMoments(clusterMoments: Array[ClusterMoment], globalCentroid: Vector) {

  /** Number of non-noise clusters actually present in the labelling — NOT the configured `k`,
   *  which an algorithm may undershoot by emptying a cluster. */
  def numClusters: Int = clusterMoments.length

  /** Total mass of the non-noise points; equals their COUNT on unweighted input. */
  def totalWeight: Double = clusterMoments.foldLeft(0.0)(_ + _.clusterWeight)
}

/** Moments of one cluster. Sums are weighted: a point of weight `w` counts as `w` copies of
 *  itself (weighting == duplication, [[clustering.core.Weights]]); on unweighted input they're
 *  the plain textbook sums.
 *
 *  @param centroid                   weighted mean of the cluster's points
 *  @param clusterWeight              Σ w — the cluster's mass (its size, unweighted)
 *  @param sumOfWeightedDistances     Σ w·d(x, centroid)
 *  @param sumOfWeightedSquaredDistances Σ w·d(x, centroid)²
 */
final case class ClusterMoment(
  label: Int,
  centroid: Vector,
  clusterWeight: Double,
  sumOfWeightedDistances: Double,
  sumOfWeightedSquaredDistances: Double
) {
  /** Dispersion S_i: the mean distance of the cluster's points to its centroid. */
  def meanDistance: Double = if (clusterWeight <= 0.0) 0.0 else sumOfWeightedDistances / clusterWeight
}

object ClusterMoments {

  /** Label of a noise point; excluded from every index, exactly as it is from the silhouette. */
  private val NoiseLabel = -1

  private val CentroidDistanceColumn = "__centroidDistance"

  /** Two passes over the labelled data: centroids and masses first, then the distance sums
   *  against those centroids.
   *
   *  The labelled frame is NOT cached despite the double scan: caching it would store a second
   *  copy of `features` ([[clustering.core.Clusterer.fit]]'s cache rule), so the labelling UDF
   *  just runs twice instead. */
  def compute(model: Model, data: DataFrame, distanceMetric: DistanceMetric): ClusterMoments = {
    val sparkSession = data.sparkSession

    val labeledData = model.assignClusters(data)
      .select(
        col(Columns.Prediction).cast(IntegerType).as(Columns.Prediction),
        col(Columns.Features),
        Weights.column(data).as(Columns.Weight))
      .filter(col(Columns.Prediction) =!= NoiseLabel)

    // Pass 1 — centroid and mass per cluster. Summarizer.mean is weight-aware natively.
    //
    // Always the arithmetic weighted mean (scikit-learn's convention), regardless of
    // `distanceMetric` — NOT always the centre the algorithm optimised (spherical geometry,
    // k-medoids, manhattan all optimise a different centre), which inflates the dispersion terms.
    // Deliberate for now — see algorithm_selection.md.
    val computedCentroids: Array[(Int, Vector, Double)] = labeledData
      .groupBy(col(Columns.Prediction))
      .agg(
        Summarizer.mean(col(Columns.Features), col(Columns.Weight)).as("centroid"),
        sum(col(Columns.Weight)).as("mass"))
      .collect()
      .map(row => (row.getInt(0), row.getAs[Vector]("centroid"), row.getDouble(2)))
      .sortBy(_._1)

    if (computedCentroids.isEmpty) return ClusterMoments(Array.empty, Vectors.zeros(0))

    // Broadcast RAW COORDINATES keyed by label: unpacked once, so the per-row scan runs the
    // same array kernel both engines use.
    val broadcastedCoordinates = sparkSession.sparkContext.broadcast(
      computedCentroids.map { case (label, centroidVector, _) => label -> centroidVector.toArray }.toMap)

    val distanceToCentroidUDF = udf { (label: Int, features: Vector) =>
      val centroid = broadcastedCoordinates.value.getOrElse(
        label,
        throw new IllegalStateException(
          s"pass 2 saw cluster $label, which pass 1 did not produce: the labelling is not stable " +
            "across scans, so these moments would be computed against the wrong centroids"))
      distanceMetric.compute(features.toArray, centroid)
    }

    // Pass 2 — Σ w·d and Σ w·d² per cluster. Distance materialised in its own projection first
    // so the UDF runs once per row, not once per aggregate mentioning it.
    val distanceSumsByLabel: Map[Int, (Double, Double)] =
      try {
        labeledData
          .withColumn(CentroidDistanceColumn, distanceToCentroidUDF(col(Columns.Prediction), col(Columns.Features)))
          .groupBy(col(Columns.Prediction))
          .agg(
            sum(col(Columns.Weight) * col(CentroidDistanceColumn)).as("sumOfDistances"),
            sum(col(Columns.Weight) * col(CentroidDistanceColumn) * col(CentroidDistanceColumn)).as("sumOfSquaredDistances"))
          .collect()
          .map(row => row.getInt(0) -> (row.getDouble(1), row.getDouble(2)))
          .toMap
      } finally broadcastedCoordinates.destroy()

    val clusterMomentsArray = computedCentroids.map { case (label, centroid, clusterMass) =>
      // A cluster pass 1 found but pass 2 did not is the mirror of the guard inside the UDF, and
      // the more dangerous half: defaulting to (0, 0) would report that cluster as having ZERO
      // dispersion, which deflates DB and inflates CH with nothing in the log to show for it.
      val (sumOfDistances, sumOfSquaredDistances) = distanceSumsByLabel.getOrElse(
        label,
        throw new IllegalStateException(
          s"cluster $label vanished between the two moment passes: the labelling is not stable " +
            "across scans"))
      ClusterMoment(label, centroid, clusterMass, sumOfDistances, sumOfSquaredDistances)
    }

    ClusterMoments(clusterMomentsArray, computeGlobalCentroid(clusterMomentsArray))
  }

  /** Weighted mean of the cluster centroids — exactly the weighted mean of the underlying
   *  points, since each centroid is already its cluster's weighted mean, so no extra pass. */
  private def computeGlobalCentroid(clusterMoments: Array[ClusterMoment]): Vector = {
    val numberOfDimensions = clusterMoments.head.centroid.size
    val coordinateSums = Array.fill(numberOfDimensions)(0.0)
    var totalGlobalWeight = 0.0

    clusterMoments.foreach { cluster =>
      val centroidCoordinates = cluster.centroid.toArray
      var dimensionIndex = 0
      while (dimensionIndex < numberOfDimensions) {
        coordinateSums(dimensionIndex) += cluster.clusterWeight * centroidCoordinates(dimensionIndex)
        dimensionIndex += 1
      }
      totalGlobalWeight += cluster.clusterWeight
    }

    if (totalGlobalWeight > 0.0) {
      var dimensionIndex = 0
      while (dimensionIndex < numberOfDimensions) {
        coordinateSums(dimensionIndex) /= totalGlobalWeight
        dimensionIndex += 1
      }
    }

    Vectors.dense(coordinateSums)
  }
}
