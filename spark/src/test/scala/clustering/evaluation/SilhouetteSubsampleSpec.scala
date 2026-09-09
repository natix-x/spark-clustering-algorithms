package clustering.evaluation

import clustering.core.{Columns, Weights}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

/** How the silhouette's subsample is drawn.
 *
 *  The silhouette is the one metric that cannot see the whole dataset, so what it sees has to
 *  stand for the whole dataset. On a WEIGHTED (reduced) input a uniform row sample does not: a
 *  row carrying most of the mass is kept or dropped whole, and the metric then describes a
 *  different population each run. These tests pin the mass-proportional draw that replaces it.
 *
 *  `knownTotalMass` is total WEIGHT on both branches — on unweighted input that is the row count.
 *  It is passed here so the tests exercise the same single-pass path the benchmark takes.
 */
class SilhouetteSubsampleSpec extends AnyFunSuite with BeforeAndAfterAll {

  private var spark: SparkSession = _

  override def beforeAll(): Unit = {
    spark = SparkSession.builder()
      .master("local[2]")
      .appName("silhouette-subsample-spec")
      .config("spark.ui.enabled", "false")
      .config("spark.sql.shuffle.partitions", "4")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
  }

  override def afterAll(): Unit = if (spark != null) spark.stop()

  private def weighted(rows: Seq[(Vector, Double)], partitions: Int = 2): DataFrame = {
    val session = spark
    import session.implicits._
    rows.toDF(Columns.Features, Columns.Weight).repartition(partitions)
  }

  private def unweighted(rows: Seq[Vector], partitions: Int = 2): DataFrame = {
    val session = spark
    import session.implicits._
    rows.map(Tuple1.apply).toDF(Columns.Features).repartition(partitions)
  }

  private def drawnPoints(sample: DataFrame): Seq[Double] =
    sample.collect().map(_.getAs[Vector](Columns.Features)(0)).sorted.toSeq

  /** `weighted` is a REQUIRED argument of `subsample` — the evaluator projects a `weight` column
   *  onto every frame, so sniffing the column would answer "weighted" for all of them and the
   *  unweighted branch would be unreachable. These tests pass the honest answer for the frame
   *  they built, which is what the evaluator passes for the frame the user supplied. */
  private def subsample(data: DataFrame, sampleSize: Int, seed: Long, mass: Double): DataFrame =
    SilhouetteEvaluator.subsample(data, sampleSize, seed, Some(mass), Weights.isWeighted(data))

  /** One row holds 99.8% of the mass. Uniform ROW sampling would give it the same 1-in-3 chance
   *  as either light row; mass-proportional sampling owes it ~99.8 of the 100 points, and the
   *  deterministic integer part of the draw guarantees at least 99 of them. */
  test("a heavy row gets a share of the sample proportional to its mass") {
    val heavy = Vectors.dense(100.0): Vector
    val data = weighted(Seq(
      Vectors.dense(0.0) -> 1.0,
      Vectors.dense(1.0) -> 1.0,
      heavy              -> 998.0))

    val drawn = subsample(data, sampleSize = 100, seed = 7L, mass = 1000.0).collect()
    val heavyCopies = drawn.count(_.getAs[Vector](Columns.Features) == heavy)

    assert(heavyCopies >= 99 && heavyCopies <= 100, s"expected ~99.8 copies, got $heavyCopies")
    assert(drawn.length >= 99, s"expected ~100 points, not ~3 rows, got ${drawn.length}")
  }

  /** Draws are unit-weighted: the sample IS the population it stands for, so re-applying the
   *  original weights on top would count the same mass twice. */
  test("every draw comes back with weight 1.0") {
    val data = weighted(Seq(
      Vectors.dense(0.0) -> 5.0,
      Vectors.dense(1.0) -> 95.0))

    val sample = subsample(data, sampleSize = 20, seed = 1L, mass = 100.0)
    assert(sample.select(Columns.Weight).distinct().collect().map(_.getDouble(0)).toSet == Set(1.0))
    assert(math.abs(Weights.totalMass(sample) - sample.count()) < 1e-9)
  }

  /** Unbiasedness is what makes the draw usable at all: E[copies] = w · sampleSize / totalMass,
   *  so the sample's size lands on sampleSize rather than merely near it. */
  test("the sample size concentrates on sampleSize") {
    val rows = (1 to 50).map(i => (Vectors.dense(i.toDouble): Vector, (i % 7 + 1).toDouble))
    val mass = rows.map(_._2).sum

    val drawn = subsample(weighted(rows), sampleSize = 60, seed = 3L, mass = mass).count()

    assert(mass > 60, "test setup: the input must actually need subsampling")
    // Only the fractional parts are random, one per row, so the total cannot drift far.
    assert(math.abs(drawn - 60) <= 10, s"expected ~60 points, got $drawn")
  }

  /** A frame whose mass already fits in the budget is passed through untouched — no draw, no
   *  rewritten weights, so a small weighted input is evaluated exactly. */
  test("input lighter than the budget is returned unchanged") {
    val data = weighted(Seq(
      Vectors.dense(0.0) -> 3.0,
      Vectors.dense(1.0) -> 4.0))

    val sample = subsample(data, sampleSize = 100, seed = 1L, mass = 7.0)
    assert(sample.count() == 2)
    assert(Weights.totalMass(sample) == 7.0, "weights must survive when no sampling happens")
  }

  /** Rows and points are the same thing on unweighted input, so keeping each row with
   *  probability `sampleSize / n` already samples the right population. No row is drawn twice:
   *  the decision is a filter, one draw per row. */
  test("unweighted input keeps each row at most once, at the target rate") {
    val data = unweighted((1 to 1000).map(i => Vectors.dense(i.toDouble)))

    val sample = subsample(data, sampleSize = 100, seed = 5L, mass = 1000.0)
    val drawn  = sample.count()

    assert(drawn > 50 && drawn < 200, s"expected ~100 rows, got $drawn")
    assert(!sample.columns.contains(Columns.Weight), "no weight column must be invented")
    assert(sample.select(Columns.Features).distinct().count() == drawn, "without replacement: no duplicates")
  }

  test("unweighted input smaller than the budget is returned unchanged") {
    val data = unweighted((1 to 10).map(i => Vectors.dense(i.toDouble)))
    assert(subsample(data, sampleSize = 100, seed = 5L, mass = 10.0).count() == 10)
  }

  /** With no mass supplied the evaluator counts (unweighted) or sums (weighted) for itself — the
   *  benchmark always supplies it, `ClusteringEvaluator.evaluate` never does. */
  test("the total is derived when the caller does not supply it") {
    val rows = (1 to 1000).map(i => Vectors.dense(i.toDouble): Vector)
    val withMass    = subsample(unweighted(rows), 100, seed = 5L, mass = 1000.0)
    val withoutMass = SilhouetteEvaluator.subsample(
      unweighted(rows), 100, seed = 5L, knownTotalMass = None, weighted = false)

    assert(drawnPoints(withMass) == drawnPoints(withoutMass))
  }

  /** THE regression test for the draw.
   *
   *  `rand(seed)` and `DataFrame.sample(seed)` seed each partition with `seed + partitionIndex`,
   *  so the sample they produce is a function of the row-to-partition layout. The benchmark
   *  matrix varies node and core counts, which varies exactly that layout — so before the
   *  content-hash draw this asserted equality FAILED, and the reported silhouette moved with
   *  worker count on a bit-identical clustering. Any future change that reintroduces a
   *  position-dependent RNG breaks here first.
   */
  test("the unweighted sample is identical under different partition counts") {
    val rows = (1 to 1000).map(i => Vectors.dense(i.toDouble): Vector)

    val few  = subsample(unweighted(rows, 1),  100, seed = 5L, mass = 1000.0)
    val many = subsample(unweighted(rows, 13), 100, seed = 5L, mass = 1000.0)

    assert(drawnPoints(few) == drawnPoints(many), "the sample must not depend on the partitioning")
    assert(drawnPoints(few).nonEmpty)
  }

  test("the weighted sample is identical under different partition counts") {
    val rows = (1 to 50).map(i => (Vectors.dense(i.toDouble): Vector, (i % 7 + 1).toDouble))
    val mass = rows.map(_._2).sum

    val few  = subsample(weighted(rows, 1),  60, seed = 3L, mass = mass)
    val many = subsample(weighted(rows, 11), 60, seed = 3L, mass = mass)

    assert(drawnPoints(few) == drawnPoints(many), "the mass-proportional draw must not depend on the partitioning")
  }

  /** A different seed must actually move the sample, otherwise "deterministic" would just mean
   *  "constant" and repetitions would measure nothing. */
  test("a different seed draws a different sample") {
    val rows = (1 to 1000).map(i => Vectors.dense(i.toDouble): Vector)

    val a = subsample(unweighted(rows), 100, seed = 5L, mass = 1000.0)
    val b = subsample(unweighted(rows), 100, seed = 6L, mass = 1000.0)

    assert(drawnPoints(a) != drawnPoints(b))
  }

  /** A negative weight costs its own row and nothing else. */
  test("a negative weight drops only its own row") {
    val data = weighted(Seq(
      Vectors.dense(0.0) -> 100.0,
      Vectors.dense(1.0) -> -5.0,
      Vectors.dense(2.0) -> 100.0))

    val drawn = subsample(data, sampleSize = 40, seed = 9L, mass = 195.0).collect()

    assert(drawn.nonEmpty, "the healthy rows must still be sampled")
    assert(!drawn.exists(_.getAs[Vector](Columns.Features)(0) == 1.0), "the negative-weight row must not appear")
  }

  /** So must a NULL one, and that is not the same code path.
   *
   *  Spark's `least`/`greatest` SKIP nulls, so a NULL weight used to make `least(null, sampleSize)`
   *  return `sampleSize` — and `greatest(sampleSize, 0)` kept it — blowing that single row up into
   *  `sampleSize` copies, i.e. the whole silhouette computed on one corrupt point. `coalesce(_, 0)`
   *  is what actually pins it to zero.
   */
  test("a NULL weight drops only its own row") {
    val session = spark
    import session.implicits._
    val data = Seq(
      (Vectors.dense(0.0): Vector, java.lang.Double.valueOf(100.0)),
      (Vectors.dense(1.0): Vector, null.asInstanceOf[java.lang.Double]),
      (Vectors.dense(2.0): Vector, java.lang.Double.valueOf(100.0))
    ).toDF(Columns.Features, Columns.Weight).repartition(2)

    val drawn = subsample(data, sampleSize = 40, seed = 9L, mass = 200.0).collect()

    assert(!drawn.exists(_.getAs[Vector](Columns.Features)(0) == 1.0), "the NULL-weight row must not appear")
    assert(drawn.length <= 41, s"the NULL-weight row must not become the sample, got ${drawn.length} points")
    assert(drawn.length >= 39, s"the healthy rows must still be sampled, got ${drawn.length} points")
  }
}
