package clustering.algorithms.kmedoids

import clustering.core.{Clusterer, Columns, Weights}
import clustering.distance.{DistanceMetric, EuclideanDistance}
import clustering.utils.DriverParallelism
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.DataFrame
import org.log4s.getLogger

/** CLARA (Kaufman & Rousseeuw 1990) — run an exact solver on small samples, keep the medoid set
 *  with the lowest cost on the FULL dataset.
 *
 *  Batched: ONE sampling job, one job dispatching a solve per sample to the CLUSTER (each sample's
 *  O(sampleSize²) distance matrix lives in its own executor's heap, not the driver's), then ONE job
 *  scoring every candidate medoid set. The samples are slices of one pool, hence disjoint rather
 *  than independent — a documented deviation, see `docs/kmedoids_docs.md`.
 *
 *  @param inner driver-local solver run on each sample (`pam` | `fastpam` | `fasterpam`)
 */
class CLARA(
  val k: Int,
  val numSamples: Int = 5,
  val sampleSize: Int = 1000,
  val maxIter:  Int  = 100,
  val distance: DistanceMetric = EuclideanDistance,
  val inner: String = "pam",
  val seed: Long = 42L
) extends Clusterer {

  private val logger = getLogger
  private val localSolver = DriverLocalKMedoids.fromName(inner, k, maxIter, distance, seed)

  override def fit(data: DataFrame): KMedoidsModel = seedMedoids(data).model

  /** Also phase I of [[PAMAE]], which needs the winner's full-data cost — returning it here keeps
   *  PAMAE from paying for a second cost job. */
  private[kmedoids] def seedMedoids(data: DataFrame): CLARA.Seeding = {
    val points = Weights.withWeights(data)
    val rowCount = points.count()
    require(rowCount >= k, s"Dataset too small: n=$rowCount points but k=$k medoids requested.")

    val samples = drawSamples(points, rowCount)

    // One sample is O(sampleSize²) work with its own distance matrix, so each is dispatched as a
    // Spark task rather than run on a driver thread — the matrix lives in that task's executor
    // heap, not the driver's, and `numSamples` executors can hold one concurrently instead of one
    // driver JVM holding all of them at once. Spark's own task scheduling is the ONLY parallelism
    // layer here: `solver.fitLocal` (PAM/FastPAM/FasterPAM) internally splits its own O(n²) scan
    // across `DriverParallelism` threads too, which is correct when it runs on the driver directly,
    // but would double up with Spark's task-per-sample parallelism here and oversubscribe the
    // executor's cores — so it is pinned to 1 thread for the lifetime of this task.
    val solver = localSolver
    val candidates = points.sparkSession.sparkContext
      .parallelize(samples.toIndexedSeq.zipWithIndex, samples.length)
      .map { case (sample, i) =>
        System.setProperty(DriverParallelism.DriverThreadsProperty, "1")
        i -> solver.fitLocal(sample.features, sample.weights)
      }
      .collect()
      .sortBy(_._1)
      .map(_._2)

    val costs = MedoidCost.perMedoidSet(points, candidates.map(_.medoids), distance)

    // Ties go to the lowest sample index.
    var winner = 0
    var i  = 1
    while (i < costs.length) {
      if (costs(i) < costs(winner)) winner = i
      i += 1
    }

    logger.info(s"clara: n=$rowCount samples=${samples.length} sampleSize=${samples.head.features.length} " +
      s"inner=$inner cost=${costs(winner)}")
    CLARA.Seeding(candidates(winner), costs(winner), rowCount)
  }

  /** One pool of `numSamples · sampleSize` rows, cut into `numSamples` disjoint slices. Data that
   *  fits in a single sample is used whole — slicing it would only make every candidate worse. */
  private def drawSamples(points: DataFrame, rowCount: Long): Array[CLARA.Sample] = {
    val fitsInOneSample = rowCount <= sampleSize
    val rows =
      if (fitsInOneSample) points.collect()
      else {
        val poolSize = math.min(rowCount, numSamples.toLong * sampleSize).toInt
        points
          .sample(withReplacement = false, fraction = math.min(1.0, 2.0 * poolSize / rowCount), seed = seed)
          .take(poolSize)
      }

    val sliceCount = if (fitsInOneSample) 1 else numSamples
    val sliceSize  = rows.length / sliceCount
    require(sliceSize >= k,
      s"Sample too small: got $sliceSize points per sample, need at least k=$k " +
        s"(n=$rowCount, numSamples=$sliceCount, sampleSize=$sampleSize).")

    // A remainder shorter than `sliceCount` is dropped — every sample keeps the same size.
    Array.tabulate(sliceCount) { i =>
      val slice = rows.slice(i * sliceSize, (i + 1) * sliceSize)
      CLARA.Sample(slice.map(_.getAs[Vector](Columns.Features)), slice.map(_.getDouble(1)))
    }
  }
}

object CLARA {

  /** The winning medoid set, its cost on the entire dataset, and the row count already paid for
   *  (so [[PAMAE]] does not count a second time). */
  private[kmedoids] final case class Seeding(model: KMedoidsModel, cost: Double, rowCount: Long)

  private[kmedoids] final case class Sample(features: Array[Vector], weights: Array[Double])
}
