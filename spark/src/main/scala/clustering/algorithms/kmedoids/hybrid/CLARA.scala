package clustering.algorithms.kmedoids.hybrid

import clustering.algorithms.kmedoids.local.DriverLocalKMedoids
import clustering.algorithms.kmedoids.components.{DriverSample, MedoidCost}
import clustering.algorithms.kmedoids.KMedoidsModel
import clustering.core.{Clusterer, Columns, Weights}
import clustering.distance.{DistanceMetric, EuclideanDistance}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.{DataFrame, Row}
import org.log4s.getLogger

/** CLARA (Kaufman & Rousseeuw 1990) — run an exact solver on small samples, keep the medoid set
 *  with the lowest cost on the FULL dataset.
 *
 *  Batched: ONE sampling job drawing all samples at once, one job dispatching a solve per sample
 *  to the CLUSTER (each sample's O(sampleSize²) distance matrix lives in its own executor's heap,
 *  not the driver's), then ONE job scoring every candidate medoid set.
 *
 *  @param inner driver-local solver run on each sample (`fastpam` | `fasterpam`). This is the knob
 *               Schubert & Rousseeuw 2021 improve CLARA by: the sampling scaffold is unchanged,
 *               only the exact solver inside it gets cheaper.
 */
class CLARA(
  val k: Int,
  val numSamples: Int = 5,
  val sampleSize: Int = 1000,
  val maxIter:  Int  = 100,
  val distance: DistanceMetric = EuclideanDistance,
  val inner: String = "fastpam",
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
    // Spark task rather than solved on the driver — the matrix lives in that task's executor heap,
    // not the driver's, and `numSamples` executors can hold one concurrently instead of one driver
    // JVM holding all of them at once. Task scheduling is the only parallelism layer involved:
    // the local solvers are single-threaded, so nothing here can oversubscribe an executor's cores.
    val solver = localSolver
    val candidates = points.sparkSession.sparkContext
      .parallelize(samples.toIndexedSeq.zipWithIndex, samples.length)
      .map { case (sample, i) => i -> solver.fitLocal(sample.features, sample.weights) }
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

  /** `numSamples` INDEPENDENT samples of `sampleSize` rows each, in ONE pass over the data. Data
   *  that fits in a single sample is used whole — sampling it would only make every candidate
   *  worse.
   *
   *  Independent, therefore OVERLAPPING, as Kaufman & Rousseeuw specify: a row may be drawn into
   *  several samples. That costs no extra pass — each row runs `numSamples` independent Bernoulli
   *  trials at once, one per sample, from `numSamples` independent `rand` streams, and is emitted
   *  once per success. Sampling S times with `DataFrame.sample` would have cost S passes; this
   *  does not, which is why the disjoint-slices shortcut this method used to take was not buying
   *  anything.
   *
   *  Two properties the previous version silently lost, both of which matter on ORDERED data
   *  (Gaia by sky position, taxi trips by date), where a positional bias is a distributional bias
   *  and not just noise:
   *
   *   - the draw is uniform over the WHOLE dataset. `take(poolSize)` returns the first rows in
   *     partition order — Spark scans partitions until the count is met — so the pool came from
   *     the beginning of the input, whatever the sampling fraction said.
   *   - a sample is a random subset, not a contiguous block. Slicing one collected array into
   *     consecutive ranges made sample i a coherent chunk of the input (one month of trips, one
   *     region of sky), which stratifies by a variable nobody chose.
   *
   *  Sizes are made exact on the driver by [[DriverSample]]: each stream over-draws, and the
   *  surplus is dropped after a seeded shuffle, so truncation cannot reintroduce the
   *  partition-order bias that dropping a suffix would. */
  private def drawSamples(points: DataFrame, rowCount: Long): Array[CLARA.Sample] = {
    if (rowCount <= sampleSize) {
      val rows = points.collect()
      require(rows.length >= k,
        s"Sample too small: got ${rows.length} points, need at least k=$k")
      return Array(CLARA.sampleOf(rows))
    }

    val inclusionProbability = DriverSample.inclusionProbability(sampleSize, rowCount)

    // One independent stream per sample: `rand(seed + s)` are separate generators, so membership
    // in sample s says nothing about membership in sample s', which is exactly the independence
    // the method assumes.
    val membership = (0 until numSamples).map { s =>
      when(rand(seed + s) < inclusionProbability, lit(s)).otherwise(lit(null).cast(IntegerType))
    }

    // The Bernoulli draws are materialised by a Project BEFORE the explode: Spark rejects a
    // nondeterministic expression inside a Generate, and re-evaluating `rand` per exploded row
    // would in any case decouple the draw from the row it is supposed to describe.
    val drawn = points
      .withColumn(CLARA.MembershipColumn, array(membership: _*))
      .withColumn(CLARA.SampleIdColumn, explode(col(CLARA.MembershipColumn)))
      .filter(col(CLARA.SampleIdColumn).isNotNull)
      .select(col(CLARA.SampleIdColumn), col(Columns.Features), col(Columns.Weight))
      .collect()
      .groupBy(_.getInt(0))

    Array.tabulate(numSamples) { s =>
      val rows = drawn.getOrElse(s, Array.empty)
      require(rows.length >= k,
        s"Sample $s too small: got ${rows.length} points, need at least k=$k " +
          s"(n=$rowCount, sampleSize=$sampleSize) — raise sampleSize.")
      CLARA.sampleOf(DriverSample.takeRandom(rows, sampleSize, seed + s))
    }
  }
}

object CLARA {

  private val SampleIdColumn = "_claraSampleId"
  private val MembershipColumn = "_claraMembership"

  private def sampleOf(rows: Array[Row]): Sample =
    Sample(rows.map(_.getAs[Vector](Columns.Features)), rows.map(_.getAs[Double](Columns.Weight)))

  /** The winning medoid set, its cost on the entire dataset, and the row count already paid for
   *  (so [[PAMAE]] does not count a second time). */
  private[kmedoids] final case class Seeding(model: KMedoidsModel, cost: Double, rowCount: Long)

  private[kmedoids] final case class Sample(features: Array[Vector], weights: Array[Double])
}
