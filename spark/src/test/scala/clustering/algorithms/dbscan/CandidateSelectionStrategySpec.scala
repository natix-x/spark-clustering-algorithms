package clustering.algorithms.dbscan

import clustering.core.Columns
import clustering.distance.EuclideanDistance
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

/** Candidate selection is where the benchmark's universal accuracy-vs-cost knob (m) is
 *  spent, so the properties it must have are the ones that make an s-sweep mean anything:
 *
 *  1. **no prefix bias** — candidates must come from the whole dataset, not from the first
 *     partitions. `take(m)`/`limit(m)` is `CollectLimitExec`, which scans partitions from 0
 *     and stops early, so any strategy that closes a sampling gap with `take` covers only
 *     part of the space. That is fatal here: the real datasets are order-correlated (Gaia by
 *     sky region, TLC by time), so the s-sweep would measure file order.
 *  2. **at most m candidates** — m bounds the driver-local ε-graph, O(m²·d).
 *  3. **roughly m candidates** — a strategy that silently returns half of m makes the
 *     accuracy-vs-cost curves of the strategies incomparable.
 *
 *  The fixture is deliberately order-correlated: the first half of the rows is one far-away
 *  group, the second half another, and the DataFrame keeps that order across partitions.
 */
class CandidateSelectionStrategySpec extends AnyFunSuite with BeforeAndAfterAll {

  private var spark: SparkSession = _

  private val Half = 100

  override def beforeAll(): Unit = {
    spark = SparkSession.builder()
      .master("local[2]")
      .appName("candidate-selection-spec")
      .config("spark.ui.enabled", "false")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
  }

  override def afterAll(): Unit = if (spark != null) spark.stop()

  /** `Half` rows near x = 0, then `Half` rows near x = 1000, in that order. */
  private def orderCorrelated(): DataFrame = {
    val session = spark
    import session.implicits._
    val early = (0 until Half).map(i => Vectors.dense(i * 0.01, 0.0))
    val late  = (0 until Half).map(i => Vectors.dense(1000.0 + i * 0.01, 0.0))
    (early ++ late).map(Tuple1.apply).toDF(Columns.Features)
  }

  private val strategies = Seq(
    UniformSelection,
    LinspaceSelection,
    new KCenterSelectionStrategy(poolFactor = 3)
  )

  for (strategy <- strategies; s <- Seq(0.1, 0.3, 0.6, 0.9)) {
    test(s"${strategy.strategyName}: s=$s covers both halves, stays within m and near m") {
      val data = orderCorrelated().cache()
      val n    = 2L * Half
      val m    = math.ceil(s * n).toInt

      val candidates =
        strategy.selectCandidates(data, n, m, seed = 11L, distanceMetric = EuclideanDistance)

      assert(candidates.length <= m,
        s"${strategy.strategyName} returned ${candidates.length} candidates for m=$m — " +
          "m bounds the driver-local ε-graph")
      // Allow the per-partition rounding slack of the deterministic strategies and the
      // ±√m of a Bernoulli sample, nothing more.
      assert(candidates.length >= m - math.max(4, 2 * math.sqrt(m).toInt),
        s"${strategy.strategyName} returned only ${candidates.length} of m=$m candidates")

      val (early, late) = candidates.partition(_(0) < 500.0)
      assert(early.nonEmpty && late.nonEmpty,
        s"${strategy.strategyName} at s=$s covered only one half of the data " +
          s"(${early.length} early, ${late.length} late) — prefix bias")
    }
  }

  // Sampling at exactly m/n is Binomial(n, m/n) ≈ m ± √m, so half the seeds would undershoot
  // and the s-sweep's x axis would carry that noise; the oversample is what removes it.
  test("uniform returns exactly m, for every seed") {
    val data = orderCorrelated().cache()
    val n    = 2L * Half
    (1L to 25L).foreach { seed =>
      val m = 20
      val candidates = UniformSelection.selectCandidates(data, n, m, seed, EuclideanDistance)
      assert(candidates.length == m,
        s"uniform returned ${candidates.length} candidates for m=$m at seed=$seed")
    }
  }

  test("all strategies are reproducible for a fixed seed and partitioning") {
    val data = orderCorrelated().cache()
    strategies.foreach { strategy =>
      def run() = strategy.selectCandidates(data, 2L * Half, 40, seed = 5L, distanceMetric = EuclideanDistance)
        .map(_.toString).toSeq
      assert(run() == run(), s"${strategy.strategyName} is not reproducible")
    }
  }

  test("s = 1.0 collects every row, for every strategy") {
    val data = orderCorrelated().cache()
    val n    = 2L * Half
    strategies.foreach { strategy =>
      val all = strategy.selectCandidates(data, n, n.toInt, seed = 1L, distanceMetric = EuclideanDistance)
      assert(all.length == n, s"${strategy.strategyName} lost rows in exact mode: ${all.length} of $n")
    }
  }

  // The per-iteration scans of the k-center traversal are split across the driver's cores;
  // the split must partition the pool exactly, in ascending order, or the traversal would
  // silently skip or double-count candidates.
  test("kcenter pool slices partition the pool in ascending order") {
    Seq(1, 10, 4095, 4096, 100000).foreach { length =>
      val slices = KCenterSelectionStrategy.sliceRanges(length)
      assert(slices.nonEmpty)
      assert(slices.head._1 == 0 && slices.last._2 == length, s"slices do not cover [0, $length)")
      slices.sliding(2).foreach {
        case Seq((_, until), (from, _)) => assert(until == from, s"slices overlap or gap at $length")
        case _                          =>
      }
      assert(slices.forall { case (from, until) => from < until })
      assert(length >= 4096 || slices.length == 1, s"pool of $length should not be split")
    }
  }

  // Ties must resolve to the LOWEST index, as in the serial scan, and slices reduce in
  // ascending order — so an equal-distance right operand never wins.
  test("kcenter slice argmaxes combine with the serial tie rule") {
    assert(KCenterSelectionStrategy.pickFarther((5.0, 3), (5.0, 9)) == ((5.0, 3)))
    assert(KCenterSelectionStrategy.pickFarther((5.0, 3), (6.0, 9)) == ((6.0, 9)))
    assert(KCenterSelectionStrategy.pickFarther((Double.NegativeInfinity, -1), (1.0, 2)) == ((1.0, 2)))
    assert(KCenterSelectionStrategy.pickFarther((1.0, 2), (Double.NegativeInfinity, -1)) == ((1.0, 2)))
  }

  test("kcenter is reproducible on a pool large enough to be split across cores") {
    val session = spark
    import session.implicits._
    val n = 6000
    val data = (0 until n)
      .map(i => Vectors.dense(math.sin(i * 0.7) * 100.0, math.cos(i * 1.3) * 100.0))
      .map(Tuple1.apply).toDF(Columns.Features).cache()

    val strategy = new KCenterSelectionStrategy(poolFactor = 3)   // pool = 4500 > one slice
    def run() = strategy.selectCandidates(data, n.toLong, 1500, seed = 7L, distanceMetric = EuclideanDistance)

    val first = run()
    assert(first.length == 1500)
    assert(first.map(_.toString).toSeq == run().map(_.toString).toSeq,
      "kcenter differs between runs — the parallel scans are not order-independent")
  }

  test("fromName rejects an unknown strategy and a poolFactor below 1") {
    assert(CandidateSelectionStrategy.fromName("kcenter", 4).strategyName == "kcenter")
    intercept[IllegalArgumentException](CandidateSelectionStrategy.fromName("nope", 4))
    intercept[IllegalArgumentException](new KCenterSelectionStrategy(poolFactor = 0))
  }
}