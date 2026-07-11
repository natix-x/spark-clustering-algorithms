package clustering.algorithms.kmeans

import clustering.core.{Clusterer, Columns, NearestPrototypeModel}
import clustering.distance.{DistanceMetric, EuclideanDistance}
import clustering.utils.Convergence
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.UserDefinedFunction

class KMeans(
  val k:        Int,
  val maxIter:  Int            = 100,
  val eps:      Double         = 1e-4,
  val distance: DistanceMetric = EuclideanDistance,
  val seed:     Long           = 42L
) extends Clusterer {

  override def fit(data: DataFrame): KMeansModel = {
    val spark = data.sparkSession

    val points = data.select(col(Columns.Features)).cache()
    val n      = points.count()

    // Phase 1: Initialize centroids — random sample
    // TODO: replace with KMeans++ for better convergence
    var centroids: Array[Vector] = points
      .sample(withReplacement = false,
              fraction        = math.min(1.0, (k * 3).toDouble / n),
              seed            = seed)
      .take(k)
      .map(_.getAs[Vector](Columns.Features))

    require(centroids.length == k,
      s"Could not sample $k initial centroids — dataset too small (n=$n).")

    var iteration    = 0
    var hasConverged = false

    while (!hasConverged && iteration < maxIter) {

      // Phase 2: Broadcast centroids to all executors
      val bc = spark.sparkContext.broadcast(centroids)

      // Phase 3: Assign each point to the nearest centroid via UDF.
      // UDF is closed over the broadcast — Catalyst pipelines it with
      // the downstream groupBy without an extra shuffle.
      val dist = distance
      val assignUDF: UserDefinedFunction = udf { features: Vector =>
        NearestPrototypeModel.nearest(features, bc.value, dist)
      }

      // Phase 4: Aggregate per cluster with Summarizer.mean.
      // Summarizer is backed by an optimised Spark aggregate —
      // no manual (sumVector / count) arithmetic needed.
      val statsMap: Map[Int, Vector] = points
        .withColumn("clusterId", assignUDF(col(Columns.Features)))
        .groupBy("clusterId")
        .agg(Summarizer.mean(col(Columns.Features)).as("newCentroid"))
        .collect()
        .map(r => r.getInt(0) -> r.getAs[Vector]("newCentroid"))
        .toMap

      bc.unpersist()

      // Phase 5: Update centroids; keep the old one for empty clusters
      val nextCentroids = (0 until k).map(i => statsMap.getOrElse(i, centroids(i))).toArray

      // Phase 6: Check convergence
      hasConverged = Convergence.hasConverged(centroids, nextCentroids, eps, distance)
      centroids    = nextCentroids
      iteration   += 1
    }

    points.unpersist(blocking = false)
    new KMeansModel(centroids, distance)
  }
}