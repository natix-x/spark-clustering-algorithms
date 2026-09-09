package clustering.algorithms.kmeans

import clustering.algorithms.kmeans.hierarchical.BisectingKMeans
import clustering.core.Columns
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

import scala.util.Random

/** `select` covers the two answers the literature gives to "which leaf is split next":
 *  `cost` (highest weighted SSE — scikit-learn's default `bisecting_strategy`) and `size`
 *  (most rows — what Steinbach, Karypis & Kumar ran). They pursue different objectives, so
 *  the test's job is to show they really are different, not that one wins.
 */
class BisectingLeafSelectionSpec extends AnyFunSuite with BeforeAndAfterAll {

  private var spark: SparkSession = _

  override def beforeAll(): Unit = {
    spark = SparkSession.builder()
      .master("local[2]")
      .appName("bisecting-leaf-selection-spec")
      .config("spark.ui.enabled", "false")
      .config("spark.sql.shuffle.partitions", "4")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
  }

  override def afterAll(): Unit = if (spark != null) spark.stop()

  /** Built so the two criteria MUST disagree: a populous, tight group (many rows, little
   *  error) against a sparse, scattered one (few rows, most of the error). After the first
   *  bisection separates them, `cost` reaches for the scattered leaf and `size` for the
   *  crowded one. */
  private def lopsided: DataFrame = {
    val rng   = new Random(3L)
    val tight = Seq.fill(400)(Vectors.dense(rng.nextGaussian() * 0.3, rng.nextGaussian() * 0.3): Vector)
    val loose = Seq.fill(50)(Vectors.dense(60.0 + rng.nextGaussian() * 4.0, rng.nextGaussian() * 4.0): Vector)
    val session = spark
    import session.implicits._
    (tight ++ loose).map(v => v -> 1.0).toDF(Columns.Features, Columns.Weight).repartition(4).cache()
  }

  private def sizes(data: DataFrame, select: String): Seq[Long] = {
    val model = new BisectingKMeans(k = 3, seed = 5L, select = select).fit(data)
    model.assignClusters(data)
      .groupBy(Columns.Prediction).count()
      .collect()
      .map(r => r.getLong(1))
      .sorted
      .toSeq
  }

  test("cost and size pick different leaves, so they build different trees") {
    val data = lopsided
    val byCost = sizes(data, "cost")
    val bySize = sizes(data, "size")
    // Both are valid 3-clusterings; only the shape differs.
    assert(byCost.size == 3 && bySize.size == 3)
    assert(byCost != bySize, s"expected different trees, got $byCost for both")
    // `size` splits the 400-row group, so its largest leaf is smaller than `cost`'s, which
    // leaves that group whole and cuts the 50-row scattered one instead. This is the
    // balance-vs-error trade-off the knob exists to measure.
    assert(bySize.max < byCost.max, s"size=$bySize was not more balanced than cost=$byCost")
  }

  test("cost is the default") {
    val data = lopsided
    assert(sizes(data, "cost") == sizes(data, "COST"))
    val default = new BisectingKMeans(k = 3, seed = 5L).fit(data)
    val explicit = new BisectingKMeans(k = 3, seed = 5L, select = "cost").fit(data)
    assert(default.clusterCentroids.map(_.toArray.toSeq).toSeq ==
           explicit.clusterCentroids.map(_.toArray.toSeq).toSeq)
  }

  /** `size` reads mass, not rows, so it must obey the repo-wide invariant: clustering a row of
   *  weight w has to equal clustering w copies of it. A row-counting criterion would fail this
   *  the moment weights are unequal — it would call a 1-row leaf of weight 300 small. */
  test("size selects by mass, so weighting a leaf equals duplicating it") {
    val session = spark
    import session.implicits._

    // A leaf that is small in rows but heavy in mass, against one that is the reverse.
    val heavy = Seq(Vectors.dense(0.0, 0.0), Vectors.dense(1.0, 0.0), Vectors.dense(0.0, 1.0))
    val light = Seq.tabulate(30)(i => Vectors.dense(50.0 + i * 0.01, 0.0))

    val weighted = (heavy.map(v => (v: Vector) -> 40.0) ++ light.map(v => (v: Vector) -> 1.0))
      .toDF(Columns.Features, Columns.Weight).repartition(3).cache()
    val duplicated = (heavy.flatMap(v => Seq.fill(40)(v: Vector)) ++ light.map(v => v: Vector))
      .map(v => v -> 1.0).toDF(Columns.Features, Columns.Weight).repartition(3).cache()

    def leafMasses(df: DataFrame): Seq[Double] = {
      val model = new BisectingKMeans(k = 3, seed = 5L, select = "size").fit(df)
      model.assignClusters(df)
        .groupBy(Columns.Prediction).agg(org.apache.spark.sql.functions.sum(Columns.Weight).as("m"))
        .collect().map(_.getDouble(1)).sorted.toSeq
    }

    val a = leafMasses(weighted)
    val b = leafMasses(duplicated)
    assert(a.size == b.size)
    a.zip(b).foreach { case (x, y) => assert(math.abs(x - y) < 1e-9, s"$a vs $b") }
  }

  test("an unknown selection is rejected before any job runs") {
    val thrown = intercept[IllegalArgumentException](new BisectingKMeans(k = 3, select = "entropy"))
    assert(thrown.getMessage.contains("cost, size"))
  }
}
