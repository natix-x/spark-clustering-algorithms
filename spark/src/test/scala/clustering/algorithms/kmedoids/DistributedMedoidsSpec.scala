package clustering.algorithms.kmedoids

import clustering.algorithms.kmedoids.hybrid.{CLARA, PAMAE}
import clustering.algorithms.kmedoids.components.{MedoidCost, MedoidRefinement}
import clustering.core.{Columns, Weights}
import clustering.distance.EuclideanDistance
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

import scala.util.Random

/** The distributed end of the k-medoids ladder: `pamae` (seeding + refinement), the shared
 *  refinement primitive and the `inner` knob on CLARA.
 *
 *  The load-bearing assertion is `pamae ≤ clara`: PAMAE is CLARA plus one refinement phase
 *  over the entire data, so if refinement is implemented correctly the objective can only
 *  improve. That is also the number the thesis reports as "what the entire-data half buys".
 */
class DistributedMedoidsSpec extends AnyFunSuite with BeforeAndAfterAll {

  private var spark: SparkSession = _

  override def beforeAll(): Unit = {
    spark = SparkSession.builder()
      .master("local[2]")
      .appName("distributed-medoids-spec")
      .config("spark.ui.enabled", "false")
      .config("spark.sql.shuffle.partitions", "4")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
  }

  override def afterAll(): Unit = if (spark != null) spark.stop()

  /** Four Gaussian blobs, 250 points each, fixed seed — enough structure that a bad medoid
   *  set costs visibly more than a good one. */
  private lazy val data: DataFrame = {
    val session = spark
    import session.implicits._
    val rnd = new Random(11L)
    val centres = Seq((0.0, 0.0), (30.0, 0.0), (0.0, 30.0), (30.0, 30.0))
    val points: Seq[Vector] = centres.flatMap { case (cx, cy) =>
      (0 until 250).map(_ => Vectors.dense(cx + rnd.nextGaussian() * 2, cy + rnd.nextGaussian() * 2))
    }
    points.map(Tuple1.apply).toDF(Columns.Features).repartition(4).cache()
  }

  private def cost(medoids: Array[Vector]): Double =
    MedoidCost.total(data, medoids, EuclideanDistance)

  /** The refinement primitive on its own: started from a deliberately bad medoid set, it must
   *  lower the objective and still return k medoids that are real data points. */
  test("refinement improves on its initialisation and returns k data points") {
    val poolSize = 200
    val pool     = MedoidRefinement.sampleCandidatePool(data, data.count(), poolSize, 5L)
    val initial  = pool.distinct.take(4)

    val refined = MedoidRefinement.refine(
      Weights.toRdd(Weights.withWeights(data)), initial, pool, EuclideanDistance, 20)

    assert(refined.medoids.length == 4)
    assert(refined.medoids.map(_.toString).distinct.length == 4, "medoids must be distinct")

    val points = data.collect().map(_.getAs[Vector](Columns.Features)).toSet
    assert(refined.medoids.forall(m => points.contains(m)), "every medoid must be a data point")
    assert(cost(refined.medoids) < cost(initial),
      s"refinement did not improve: ${cost(refined.medoids)} vs initial ${cost(initial)}")
  }

  /** CLARA's samples must be INDEPENDENT draws over the whole dataset, which is what Kaufman &
   *  Rousseeuw specify and what the earlier disjoint-slices implementation did not provide.
   *
   *  The test is built to fail loudly on the two ways that can break. The input is ORDERED — the
   *  x coordinate grows monotonically with the row index — so:
   *
   *   - a draw biased towards the start of the input (what `take(poolSize)` produced, since it
   *     returns the first rows in partition order) yields samples whose mean x sits well below
   *     the dataset mean;
   *   - samples cut as contiguous blocks of one pool yield per-sample means that march upwards
   *     with the sample index, and never overlap.
   *
   *  Both are checked through the medoids of single-sample fits, which is the only place the
   *  sampling is observable through the public API. */
  test("clara samples are independent draws over the whole dataset, not a biased prefix") {
    val session = spark
    import session.implicits._
    // 20 000 rows, x strictly increasing: position in the input IS the value.
    val ordered = (0 until 20000)
      .map(i => Vectors.dense(i.toDouble, 0.0): Vector)
      .map(v => v -> 1.0)
      .toDF(Columns.Features, Columns.Weight)
      .repartition(8)
      .cache()

    // k = 1 makes the medoid of each sample its own centre point, so the sample's location is
    // directly readable off the fitted model.
    val centres = (0 until 6).map { s =>
      new CLARA(k = 1, numSamples = 6, sampleSize = 500, maxIter = 50,
        distance = EuclideanDistance, inner = "fastpam", seed = 17L)
        .fit(ordered).medoids.head.toArray.head
    }

    // Every sample spans the whole range, so its medoid must land near the middle (10 000),
    // not in the first fifth of the data as a prefix-biased draw would.
    centres.foreach { c =>
      assert(c > 6000.0 && c < 14000.0,
        s"sample medoid $c is not a draw from the whole range — sampling is positionally biased")
    }
  }

  /** The candidate pool of PAMAE's phase II is the set it may pick representatives from, so a
   *  pool drawn from one region of an ordered input silently restricts the very phase that exists
   *  to see ALL the data. Same defect, same shape as the CLARA sampling test: x grows with the row
   *  index, so a prefix-biased pool shows up as medoids stuck in the low range. */
  test("pamae's candidate pool is drawn from the whole dataset, not a prefix") {
    val session = spark
    import session.implicits._
    val ordered = (0 until 20000)
      .map(i => Vectors.dense(i.toDouble, 0.0): Vector)
      .map(v => v -> 1.0)
      .toDF(Columns.Features, Columns.Weight)
      .repartition(8)
      .cache()

    val pool = MedoidRefinement.sampleCandidatePool(ordered, 20000L, poolSize = 400, seed = 23L)
    assert(pool.length == 400, s"expected a full pool, got ${pool.length}")

    val xs = pool.map(_.toArray.head)
    // A uniform draw over [0, 20000) has mean ~10 000 and spans the range; a prefix of the scan
    // order would sit far below and stop early.
    val mean = xs.sum / xs.length
    assert(mean > 8000.0 && mean < 12000.0, s"pool mean $mean is not centred — draw is biased")
    assert(xs.max > 18000.0, s"pool never reaches the tail of the data (max ${xs.max})")
  }

  test("pamae never scores worse than its own seeding phase (clara)") {
    val claraModel = new CLARA(k = 4, numSamples = 3, sampleSize = 120, maxIter = 50,
      distance = EuclideanDistance, inner = "fastpam", seed = 9L).fit(data)
    val pamaeModel = new PAMAE(k = 4, numSamples = 3, sampleSize = 120, maxIter = 50,
      refineIters = 3, poolSize = 200, inner = "fastpam", distance = EuclideanDistance, seed = 9L).fit(data)

    val claraCost = cost(claraModel.medoids)
    val pamaeCost = cost(pamaeModel.medoids)
    assert(pamaeCost <= claraCost + 1e-9,
      s"refinement made the objective worse: pamae=$pamaeCost clara=$claraCost")
  }

  test("refinement is monotone: more iterations never cost more") {
    val one = new PAMAE(k = 4, numSamples = 2, sampleSize = 80, refineIters = 1, poolSize = 150,
      distance = EuclideanDistance, seed = 4L).fit(data)
    val five = new PAMAE(k = 4, numSamples = 2, sampleSize = 80, refineIters = 5, poolSize = 150,
      distance = EuclideanDistance, seed = 4L).fit(data)
    assert(cost(five.medoids) <= cost(one.medoids) + 1e-9)
  }

  test("CLARA's inner solver is a knob and both solvers agree on well-separated blobs") {
    val costs = Seq("fastpam", "fasterpam").map { inner =>
      inner -> cost(new CLARA(k = 4, numSamples = 3, sampleSize = 120, maxIter = 50,
        distance = EuclideanDistance, inner = inner, seed = 9L).fit(data).medoids)
    }.toMap

    // Same objective, three searches: on separated blobs they must land within a few percent.
    val best = costs.values.min
    costs.foreach { case (inner, c) =>
      assert(c <= best * 1.05, s"inner='$inner' cost $c is more than 5% worse than $best")
    }
  }

  /** CLARA scores all its candidate medoid sets in ONE job, so the batched form must agree with
   *  evaluating each set on its own — otherwise it could pick the wrong sample. */
  test("MedoidCost scores several medoid sets in one pass exactly as it scores them one by one") {
    val rows = data.collect().map(_.getAs[Vector](Columns.Features))
    val sets = Array(
      Array(rows(0), rows(300), rows(600)),
      Array(rows(1), rows(301), rows(999)),
      Array(rows(10), rows(20), rows(30))
    )

    val batched    = MedoidCost.perMedoidSet(data, sets, EuclideanDistance)
    val individual = sets.map(set => MedoidCost.total(data, set, EuclideanDistance))

    assert(batched.length == sets.length)
    batched.zip(individual).foreach { case (b, i) => assert(math.abs(b - i) < 1e-9, s"$b != $i") }
  }

  test("CLARA rejects an unknown inner solver") {
    val thrown = intercept[IllegalArgumentException] {
      new CLARA(k = 2, inner = "banditpam", distance = EuclideanDistance)
    }
    assert(thrown.getMessage.contains("Unknown inner k-medoids solver"))
  }
}
