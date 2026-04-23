package clustering.algorithms.kmedoids

import clustering.core.Clusterer
import clustering.data.{DatasetOps, Point}
import clustering.distance.{DistanceMetric, EuclideanDistance}
import clustering.utils.SparkUtils
import org.apache.spark.rdd.RDD


/** Clustering Large Applications (CLARA).
 *
 *  Runs PAM on `numSamples` random subsets of the data (each of size
 *  `sampleFraction * n`), then scores every candidate model on the
 *  full dataset and returns the best one.
 *
 *  Scales to large datasets where full PAM would be prohibitive.
 *  Quality improves with more samples and larger sample fractions.
 */
class CLARA(
  val k: Int,
  val numSamples: Int = 5,
  val sampleFraction: Double = 0.1,
  val maxIter: Int = 100,
  val distance: DistanceMetric = new EuclideanDistance()
) extends Clusterer {

  private val pam = new PAM(k, maxIter, distance)

  override def fit(data: RDD[Point]): KMedoidsModel = {
    val cached = DatasetOps.cachePoints(data)

    (0 until numSamples)
      .map { seed =>
        val sample = cached.sample(withReplacement = false, fraction = sampleFraction, seed = seed.toLong)
        val model  = pam.fit(sample)
        val cost   = evaluateCost(cached, model)
        (cost, model)
      }
      .minBy(_._1)
      ._2
  }

  private def evaluateCost(data: RDD[Point], model: KMedoidsModel): Double = {
    implicit val sc = data.sparkContext
    val bcMedoids   = SparkUtils.broadcastSafe(model.medoids)
    val cost        = data.map { p =>
      bcMedoids.value.map(m => distance.compute(m, p)).min
    }.sum()
    bcMedoids.destroy()
    cost
  }
}
