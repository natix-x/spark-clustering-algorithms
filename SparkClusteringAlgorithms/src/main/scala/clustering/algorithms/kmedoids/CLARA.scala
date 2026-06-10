package clustering.algorithms.kmedoids

import clustering.core.Clusterer
import clustering.distance.{DistanceMetric, EuclideanDistance}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.UserDefinedFunction


class CLARA(
  val k:          Int,
  val numSamples: Int          = 5,
  val sampleSize: Int          = 1000,
  val maxIter:    Int          = 100,
  val distance:   DistanceMetric = EuclideanDistance
) extends Clusterer {

  private val pam = new PAM(k, maxIter, distance)

  override def fit(data: DataFrame): KMedoidsModel = {
    val spark  = data.sparkSession
    val cached = data.select(col("features")).cache()

    var bestModel: KMedoidsModel = null
    var minCost                  = Double.MaxValue

    for (i <- 0 until numSamples) {

      // Phase 1: Sample locally — Dataset.sample + collect stays in driver memory.
      // sampleSize << total n, so this is safe.
      val fraction    = math.min(1.0, sampleSize.toDouble * 2 / cached.count())
      val sampleArray = cached
        .sample(withReplacement = false, fraction = fraction, seed = i.toLong)
        .take(sampleSize)
        .map(_.getAs[Vector]("features"))

      require(sampleArray.length >= k,
        s"Sample too small: got ${sampleArray.length} points, need at least k=$k.")

      // Phase 2: Run PAM locally on the small sample.
      // PAM is O(n²) — feasible only because sampleSize is small.
      val candidateModel = pam.fitLocal(sampleArray)

      // Phase 3: Evaluate candidate model quality on the full distributed dataset.
      val cost = evaluateCost(cached, candidateModel, spark)

      if (cost < minCost) {
        minCost = cost
        bestModel = candidateModel
      }
    }

    cached.unpersist(blocking = false)
    bestModel
  }

  /** Computes total assignment cost (sum of distances to nearest medoid)
   *  over the full dataset using broadcast + UDF — one Spark job per sample.
   */
  private def evaluateCost(
    data:  DataFrame,
    model: KMedoidsModel,
    spark: SparkSession
  ): Double = {
    import spark.implicits._

    val bc   = spark.sparkContext.broadcast(model.medoids)
    val dist = distance

    val costUDF: UserDefinedFunction = udf { features: Vector =>
      val medoids = bc.value
      var minDist = Double.MaxValue
      var j       = 0
      while (j < medoids.length) {
        val d = dist.compute(features, medoids(j))
        if (d < minDist) minDist = d
        j += 1
      }
      minDist
    }

    val totalCost = data
      .select(costUDF(col("features")).as("pointCost"))
      .agg(sum("pointCost"))
      .as[Double]
      .head()

    bc.unpersist(blocking = false)
    totalCost
  }
}
