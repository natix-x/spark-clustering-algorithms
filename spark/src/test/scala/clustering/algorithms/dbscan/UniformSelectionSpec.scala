package clustering.algorithms.dbscan

import clustering.algorithms.dbscan.components.UniformSelection
import clustering.core.Columns
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
class UniformSelectionSpec extends AnyFunSuite with BeforeAndAfterAll {

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

  for (s <- Seq(0.1, 0.3, 0.6, 0.9)) {
    test(s"uniform: s=$s covers both halves, stays within m and near m") {
      val data = orderCorrelated().cache()
      val n    = 2L * Half
      val m    = math.ceil(s * n).toInt

      val candidates =
        UniformSelection.selectCandidates(data, n, m, seed = 11L)

      assert(candidates.length <= m,
        s"uniform returned ${candidates.length} candidates for m=$m — " +
          "m bounds the driver-local ε-graph")
      // Allow the ±√m slack of a Bernoulli sample, nothing more.
      assert(candidates.length >= m - math.max(4, 2 * math.sqrt(m).toInt),
        s"uniform returned only ${candidates.length} of m=$m candidates")

      val (early, late) = candidates.partition(_(0) < 500.0)
      assert(early.nonEmpty && late.nonEmpty,
        s"uniform at s=$s covered only one half of the data " +
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
      val candidates = UniformSelection.selectCandidates(data, n, m, seed)
      assert(candidates.length == m,
        s"uniform returned ${candidates.length} candidates for m=$m at seed=$seed")
    }
  }

  test("uniform is reproducible for a fixed seed and partitioning") {
    val data = orderCorrelated().cache()
    def run(): Seq[String] = UniformSelection.selectCandidates(data, 2L * Half, 40, seed = 5L)
      .map(_.toString).toSeq
    assert(run() == run(), "uniform is not reproducible")
  }

  test("s = 1.0 collects every row") {
    val data = orderCorrelated().cache()
    val n    = 2L * Half
    val all = UniformSelection.selectCandidates(data, n, n.toInt, seed = 1L)
    assert(all.length == n, s"uniform lost rows in exact mode: ${all.length} of $n")
  }

  test("fromName resolves uniform and rejects an unknown strategy") {
    assert(UniformSelection.fromName("uniform").strategyName == "uniform")
    intercept[IllegalArgumentException](UniformSelection.fromName("nope"))
  }
}
