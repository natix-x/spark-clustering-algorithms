package clustering.algorithms.kmeans

import clustering.core.{Columns, Geometry, NearestPrototypeModel, Weights}
import clustering.distance.DistanceMetric
import clustering.utils.Convergence
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.storage.StorageLevel

/** Lloyd's iteration, extracted so that both [[KMeans]] and [[BreathingKMeans]] run the SAME
 *  loop — the latter only wraps it in an outer add/remove cycle, which is what makes
 *  `refine: breathing` a knob on one algorithm rather than a second algorithm.
 *
 *  Operates on ALREADY PREPARED data: `[features, weight]`, projected into the geometry's space
 *  (see [[clustering.core.Weights]] and [[Geometry]]). The number of clusters is
 *  `initial.length`, not a field, so the same loop serves k and k+m centroids.
 *
 *  Breathing-only statistics (mass/error/utility per centroid) live in [[BreathingKMeans]]
 *  itself, not here — this object stays the shared kernel every centroid-based algorithm
 *  (`KMeans`, `BreathingKMeans`, `BisectingKMeans`) uses, not a dumping ground for one knob's
 *  private math.
 */
private[kmeans] object LloydKMeans {

  final case class LloydContext(preparedPoints: DataFrame, fitDistance: DistanceMetric, initialCentroids: Array[Vector])

  def initialize(
    data:         DataFrame,
    geometry:     Geometry,
    k:            Int,
    seed:         Long,
    storageLevel: StorageLevel
  ): LloydContext = {
    val fitDistance = geometry.fitDistance
    val preparedPoints = geometry.prepare(Weights.withWeights(data)).persist(storageLevel)
    val n = preparedPoints.count()

    val initialCentroids = sampleInitialCentroids(preparedPoints, k, seed)
    require(initialCentroids.length == k,
      s"Could not sample $k initialCentroids centroids — dataset too small (n=$n).")

    LloydContext(preparedPoints, fitDistance, initialCentroids)
  }

  def run(
     points:   DataFrame,
     initialCentroids:  Array[Vector],
     fitDistance: DistanceMetric,
     geometry: Geometry,
     maxIter:  Int,
     eps:      Double
  ): Array[Vector] = {
    require(initialCentroids.nonEmpty, "Lloyd.run: initial centroids must not be empty")
    val sc = points.sparkSession.sparkContext
    val kk = initialCentroids.length

    var centroids = initialCentroids
    var iteration = 0
    var hasConverged = false

    while (!hasConverged && iteration < maxIter) {
      val bc   = sc.broadcast(centroids)
      val dist = fitDistance
      val assignUDF = udf { features: Vector =>
        NearestPrototypeModel.nearest(features, bc.value, dist)
      }

      // Weighted mean: a point of weight w counts as w points, so a coreset yields the centroid
      // of the data it stands for. Summarizer supports weights natively.
      val newCentroidsByCluster: Map[Int, Vector] = points
        .withColumn("clusterId", assignUDF(col(Columns.Features)))
        .groupBy("clusterId")
        .agg(Summarizer.mean(col(Columns.Features), col(Columns.Weight)).as("newCentroid"))
        .collect()
        .map(r => r.getInt(0) -> r.getAs[Vector]("newCentroid"))
        .toMap

      bc.unpersist()

      val newCentroids = Array.tabulate(kk) { i =>
        newCentroidsByCluster.get(i) match {
          case Some(newCentroid) => geometry.project(newCentroid)
          case None              => centroids(i) // fallback for empty clusters
        }
      }

      hasConverged = Convergence.hasConverged(centroids, newCentroids, eps, fitDistance)
      centroids    = newCentroids
      iteration   += 1
    }

    centroids
  }

  /** Seeded weighted sample of `count` distinct-ish starting centroids, taken from prepared data.
   *  Kept here so [[KMeans]] and [[BreathingKMeans]] initialise identically.
   *
   *  A-Res weighted reservoir sampling: score = `rand(seed)^(1/weight)`, take the `count`
   *  highest. With `weight = 1.0` everywhere this collapses to plain uniform sampling by
   *  `rand(seed)`, so unweighted runs are unaffected — one code path covers both, consistent
   *  with the "weighting == duplication" invariant (see [[clustering.core.Weights]]).
   */
  def sampleInitialCentroids(points: DataFrame, count: Int, seed: Long): Array[Vector] =
    points
      .withColumn("_key", pow(rand(seed), lit(1.0) / col(Columns.Weight)))
      .orderBy(desc("_key"))
      .limit(count)
      .collect()
      .map(_.getAs[Vector](Columns.Features))
}
