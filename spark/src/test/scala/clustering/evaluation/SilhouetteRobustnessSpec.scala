package clustering.evaluation

import clustering.core.{Columns, Model}
import clustering.distance.EuclideanDistance
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

/** What the silhouette does with input it cannot take at face value, and what it reports about
 *  the sample it actually scored.
 *
 *  `SilhouetteSpec` pins the arithmetic on clean data and `SilhouetteSubsampleSpec` pins the draw;
 *  neither covers the seam between them — the 5-argument `measure`, which is the ONLY entry point
 *  the benchmark uses. Everything here is a defect that produced a plausible-looking number rather
 *  than an error, which is what makes it worth a test: a wrong silhouette is indistinguishable
 *  from a differently-clustered dataset.
 */
class SilhouetteRobustnessSpec extends AnyFunSuite with BeforeAndAfterAll {

  private var spark: SparkSession = _

  override def beforeAll(): Unit = {
    spark = SparkSession.builder()
      .master("local[2]")
      .appName("silhouette-robustness-spec")
      .config("spark.ui.enabled", "false")
      .config("spark.sql.shuffle.partitions", "4")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
  }

  override def afterAll(): Unit = if (spark != null) spark.stop()

  private object LabelColumnModel extends Model {
    val LabelCol = "__label"
    override def assignClusters(data: DataFrame): DataFrame =
      data.withColumn(Columns.Prediction, col(LabelCol))
  }

  private def frame(rows: Seq[(Vector, Int)]): DataFrame = {
    val session = spark
    import session.implicits._
    rows.toDF(Columns.Features, LabelColumnModel.LabelCol).repartition(3)
  }

  private def weightedFrame(rows: Seq[(Vector, Int, java.lang.Double)]): DataFrame = {
    val session = spark
    import session.implicits._
    rows.toDF(Columns.Features, LabelColumnModel.LabelCol, Columns.Weight).repartition(3)
  }

  private def measure(
    data: DataFrame,
    sampleSize: Option[Int] = None,
    seed: Long = 0L,
    population: Option[SilhouetteEvaluator.Population] = None
  ): SilhouetteEvaluator.Outcome =
    new SilhouetteEvaluator(EuclideanDistance)
      .measure(LabelColumnModel, data, sampleSize, seed, population)

  private def assertClose(actual: Double, expected: Double): Unit =
    assert(math.abs(actual - expected) < 1e-9, s"expected $expected, got $actual")

  /** The reference labelling every test below degrades: {0, 2} and {10, 12}, mean 79/99. */
  private val CleanScore = 79.0 / 99.0
  private val CleanRows = Seq(
    (Vectors.dense(0.0): Vector)  -> 0,
    (Vectors.dense(2.0): Vector)  -> 0,
    (Vectors.dense(10.0): Vector) -> 1,
    (Vectors.dense(12.0): Vector) -> 1)

  // ── corrupt input: one bad row must cost its own row and nothing else ──────────────────────

  /** A NaN coordinate makes its cluster's distance SUM NaN for every other point, and `mean < b`
   *  is false for NaN — so the poisoned cluster was silently skipped in the `b` minimum and every
   *  point of the neighbouring cluster took its `b` from a cluster further away. The score went
   *  UP, toward 1, with nothing in the output to show for it. Here cluster 1 is cluster 0's
   *  nearest neighbour, so a surviving NaN would inflate both of cluster 0's scores.
   */
  test("a NaN coordinate costs its own row, not every other point's b") {
    val outcome = measure(frame(CleanRows :+ ((Vectors.dense(Double.NaN): Vector), 1)))

    assertClose(outcome.score, CleanScore)
    assert(outcome.scoredPoints == 4)
    assert(outcome.unscoredPoints == 1, "the NaN row must be reported, not silently absorbed")
  }

  test("an infinite coordinate is refused the same way") {
    val outcome = measure(frame(CleanRows :+ ((Vectors.dense(Double.PositiveInfinity): Vector), 1)))
    assertClose(outcome.score, CleanScore)
    assert(outcome.unscoredPoints == 1)
  }

  /** The NULL weight used to reach `Row.getDouble` as an NPE — but only on the paths that do NOT
   *  draw (`sampleSize = None`, or an input already lighter than the budget), because the
   *  `coalesce` guard lived inside the weighted branch of `subsample`. A corrupt weight then
   *  failed the whole run and the result file said `status: failed`, not "one bad row".
   */
  test("a NULL weight costs its own row on the no-draw path") {
    val rows = CleanRows.map { case (v, l) => (v, l, java.lang.Double.valueOf(1.0)) } :+
      (((Vectors.dense(5.0): Vector), 0, null.asInstanceOf[java.lang.Double]))

    val outcome = measure(weightedFrame(rows))
    assertClose(outcome.score, CleanScore)
    assert(outcome.scoredPoints == 4)
  }

  /** Same path, worse symptom: a negative weight makes its cluster's mean distance negative,
   *  which wins the `b` minimum for EVERY point in the frame and drags the whole score toward -1.
   */
  test("a negative weight costs its own row on the no-draw path") {
    val rows = CleanRows.map { case (v, l) => (v, l, java.lang.Double.valueOf(1.0)) } :+
      (((Vectors.dense(5.0): Vector), 0, java.lang.Double.valueOf(-5.0)))

    val outcome = measure(weightedFrame(rows))
    assertClose(outcome.score, CleanScore)
    assert(outcome.scoredPoints == 4)
  }

  test("a NaN weight costs its own row on the no-draw path") {
    val rows = CleanRows.map { case (v, l) => (v, l, java.lang.Double.valueOf(1.0)) } :+
      (((Vectors.dense(5.0): Vector), 0, java.lang.Double.valueOf(Double.NaN)))

    // NaN compares GREATER than everything in Spark SQL, so a plain `weight > 0` filter keeps it.
    val outcome = measure(weightedFrame(rows))
    assertClose(outcome.score, CleanScore)
    assert(outcome.scoredPoints == 4)
  }

  // ── singleton vs under-drawn: the same shape, opposite meanings ────────────────────────────

  /** A cluster whose FULL mass is 1 has no neighbours and scores 0 by Rousseeuw's definition. */
  test("a genuinely singleton cluster still scores 0") {
    val rows = Seq(
      (Vectors.dense(0.0): Vector)   -> 0,
      (Vectors.dense(2.0): Vector)   -> 0,
      (Vectors.dense(100.0): Vector) -> 1)
    val population = SilhouetteEvaluator.Population(Map(0 -> 2.0, 1 -> 1.0))

    val outcome = measure(frame(rows), population = Some(population))
    assertClose(outcome.score, (0.98 + 96.0 / 98.0) / 3.0)
    assert(outcome.unscoredPoints == 0, "a true singleton is scored, not skipped")
  }

  /** The case the singleton convention used to swallow. Cluster 2 holds 1000 points in the full
   *  data but only ONE reached the sample, so its `a` is unestimable. Scoring it 0 injects a hard
   *  zero into the mean for a cluster that may be perfectly separated — a discrete, k-dependent
   *  downward bias on top of the min-of-noisy-`b` one, and on a skewed dataset at k = 100 it fires
   *  for many clusters at once. The point is excluded and counted instead.
   */
  test("a cluster the sample under-drew is excluded, not scored 0") {
    val rows = CleanRows :+ ((Vectors.dense(1000.0): Vector), 2)
    val population = SilhouetteEvaluator.Population(Map(0 -> 2.0, 1 -> 2.0, 2 -> 1000.0))

    val outcome = measure(frame(rows), population = Some(population))

    assertClose(outcome.score, CleanScore)   // exactly the four clean points, no injected zero
    assert(outcome.scoredPoints == 4)
    assert(outcome.unscoredPoints == 1)
    assert(outcome.sampleClusters == 3, "the under-drawn cluster is still visible to b")
  }

  /** Without the full-data masses the two cases are indistinguishable, and the evaluator falls
   *  back to the sample's own masses — i.e. the old behaviour. Pinned so the fallback stays a
   *  deliberate, documented degradation rather than an accident. */
  test("without a population the under-drawn cluster falls back to the singleton rule") {
    val outcome = measure(frame(CleanRows :+ ((Vectors.dense(1000.0): Vector), 2)))
    assert(outcome.scoredPoints == 5)
    assert(outcome.unscoredPoints == 0)
  }

  // ── weights ───────────────────────────────────────────────────────────────────────────────

  /** `a` divides by `mass − min(1, w)`, not `mass − 1`.
   *
   *  Scaling every weight by the same factor cancels out of both `a` and `b`, so a uniformly
   *  weighted frame must score exactly like the unweighted one. With a flat `mass − 1` and
   *  FRACTIONAL weights it did not: two rows of weight 0.6 at distance 1 gave
   *  `a = 0.6 / (1.2 − 1) = 3`, a 5× inflation of a mean distance of 1. Only reachable on the
   *  no-draw path (drawn samples are unit-weighted), which is exactly where a small coreset lands.
   */
  test("uniform fractional weights score like unweighted input") {
    val rows = Seq(
      (Vectors.dense(0.0): Vector)   -> 0,
      (Vectors.dense(1.0): Vector)   -> 0,
      (Vectors.dense(100.0): Vector) -> 1,
      (Vectors.dense(101.0): Vector) -> 1)

    val plain = measure(frame(rows)).score
    val light = measure(weightedFrame(rows.map { case (v, l) => (v, l, java.lang.Double.valueOf(0.6)) })).score

    assertClose(light, plain)
  }

  /** The other half of the same `min(1, w)`: for `w ≥ 1` the removed share is exactly 1, so
   *  duplication semantics are untouched. A uniformly weight-3 frame is NOT the unweighted frame
   *  (each copy has two more copies of itself at distance 0 in its cluster) — it is the 3×
   *  duplicated one, and that is what it must equal. */
  test("uniform integer weights still equal duplication, not the unweighted frame") {
    val rows = Seq(
      (Vectors.dense(0.0): Vector)   -> 0,
      (Vectors.dense(1.0): Vector)   -> 0,
      (Vectors.dense(100.0): Vector) -> 1,
      (Vectors.dense(101.0): Vector) -> 1)

    val heavy      = measure(weightedFrame(rows.map { case (v, l) => (v, l, java.lang.Double.valueOf(3.0)) })).score
    val duplicated = measure(frame(rows.flatMap { case (v, l) => Seq.fill(3)((v, l)) })).score

    assertClose(heavy, duplicated)
    assert(math.abs(heavy - measure(frame(rows)).score) > 1e-3,
      "test setup: duplication must actually differ from the unweighted frame here")
  }

  // ── the draw → score seam, which no test used to cross ─────────────────────────────────────

  /** The 5-argument `measure` end to end: subsample, collect, broadcast, score. */
  test("subsampled evaluation scores about sampleSize points and reports how many") {
    val rows = (1 to 600).map(i => ((Vectors.dense(i.toDouble): Vector), if (i <= 300) 0 else 1))
    val population = SilhouetteEvaluator.Population(Map(0 -> 300.0, 1 -> 300.0))

    val outcome = measure(frame(rows), sampleSize = Some(100), seed = 11L, population = Some(population))

    assert(outcome.scoredPoints > 60 && outcome.scoredPoints < 150,
      s"expected ~100 scored points, got ${outcome.scoredPoints}")
    assert(outcome.sampleClusters == 2)
    assert(outcome.unscoredPoints == 0)
    assert(outcome.score > 0.5, s"two well-separated blocks must score high, got ${outcome.score}")
  }

  /** The reported count is the ACHIEVED sample size, not the configured one — the draw is
   *  Binomial around `sampleSize`, so a result that assumed equality would be quietly wrong. */
  test("the draw is deterministic in the seed and reported, not assumed") {
    val rows = (1 to 600).map(i => ((Vectors.dense(i.toDouble): Vector), if (i <= 300) 0 else 1))
    val population = Some(SilhouetteEvaluator.Population(Map(0 -> 300.0, 1 -> 300.0)))

    val a = measure(frame(rows), Some(100), seed = 11L, population)
    val b = measure(frame(rows), Some(100), seed = 11L, population)
    val c = measure(frame(rows), Some(100), seed = 12L, population)

    assert(a == b, "same seed, same sample, same number")
    assert(a.score != c.score, "a different seed must move the estimate — that spread IS the variance")
  }

  /** An input that already fits the budget must not be drawn at all: the early-out was
   *  unreachable while the evaluator's own projection made every frame look weighted. */
  test("an input lighter than the budget is scored exactly") {
    val outcome = measure(frame(CleanRows), sampleSize = Some(10000), seed = 3L,
      population = Some(SilhouetteEvaluator.Population(Map(0 -> 2.0, 1 -> 2.0))))

    assertClose(outcome.score, CleanScore)
    assert(outcome.scoredPoints == 4)
  }

  /** Noise is dropped BEFORE the draw, so `sampleSize` counts scored points on every algorithm —
   *  and the reported score therefore describes the clustered part only. */
  test("noise is outside both the draw and the counts") {
    val rows = CleanRows ++ Seq(
      (Vectors.dense(500.0): Vector)  -> -1,
      (Vectors.dense(-500.0): Vector) -> -1)

    val outcome = measure(frame(rows))
    assertClose(outcome.score, CleanScore)
    assert(outcome.scoredPoints == 4)
    assert(outcome.sampleClusters == 2)
  }

  test("fewer than two clusters yields 0.0 and says so") {
    val outcome = measure(frame(Seq(
      (Vectors.dense(0.0): Vector) -> 0,
      (Vectors.dense(2.0): Vector) -> 0)))

    assert(outcome.score == 0.0)
    assert(outcome.scoredPoints == 0)
    assert(outcome.sampleClusters == 1)
  }
}
