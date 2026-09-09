package clustering.evaluation

import clustering.core.{Columns, Model}
import clustering.distance.{CosineDistance, EuclideanDistance}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

/** The silhouette, pinned against values computed BY HAND from Rousseeuw's definition.
 *
 *  It is the metric the thesis quotes most and the only O(n²) one, so both its arithmetic and
 *  its degenerate cases are fixed here rather than left to whatever the implementation happens
 *  to return. The labelling is supplied by the test (a `Model` reading a label column), so what
 *  is under test is the evaluator alone.
 */
class SilhouetteSpec extends AnyFunSuite with BeforeAndAfterAll {

  private var spark: SparkSession = _

  override def beforeAll(): Unit = {
    spark = SparkSession.builder()
      .master("local[2]")
      .appName("silhouette-spec")
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

  private def weightedFrame(rows: Seq[(Vector, Int, Double)]): DataFrame = {
    val session = spark
    import session.implicits._
    rows.toDF(Columns.Features, LabelColumnModel.LabelCol, Columns.Weight).repartition(3)
  }

  private def silhouette(data: DataFrame, metric: clustering.distance.DistanceMetric = EuclideanDistance): Double =
    new SilhouetteEvaluator(metric).evaluate(LabelColumnModel, data)

  private def assertClose(actual: Double, expected: Double): Unit =
    assert(math.abs(actual - expected) < 1e-9, s"expected $expected, got $actual")

  /** {0, 2} and {10, 12}. a = 2 for every point;
   *  b = 11, 9, 9, 11  ->  s = 9/11, 7/9, 7/9, 9/11, mean 79/99. */
  test("mean silhouette matches the hand-computed value") {
    assertClose(silhouette(frame(Seq(
      Vectors.dense(0.0)  -> 0,
      Vectors.dense(2.0)  -> 0,
      Vectors.dense(10.0) -> 1,
      Vectors.dense(12.0) -> 1))), 79.0 / 99.0)
  }

  /** The case that used to need a synthetic row id: two DISTINCT rows sharing coordinates.
   *
   *  Cluster {0, 0, 2}, plus {10}. For either point at 0, `a` must average over the OTHER two
   *  members — the twin at distance 0 and the point at distance 2 — giving 1, not the 2 that
   *  dropping every coordinate-equal row would give. Hand: s = 0.9, 0.9, 0.75, 0 -> 0.6375.
   */
  test("a coordinate-duplicate neighbour counts, only the point itself is excluded") {
    assertClose(silhouette(frame(Seq(
      Vectors.dense(0.0)  -> 0,
      Vectors.dense(0.0)  -> 0,
      Vectors.dense(2.0)  -> 0,
      Vectors.dense(10.0) -> 1))), 0.6375)
  }

  /** Rousseeuw: a point alone in its cluster scores 0, NOT 1. Treating its undefined `a` as 0
   *  would reward exactly the labellings (stray singletons out of a density or medoid run) that
   *  the score is supposed to punish. Hand: 0.98, 96/98, 0. */
  test("a singleton cluster contributes 0, not 1") {
    assertClose(silhouette(frame(Seq(
      Vectors.dense(0.0)   -> 0,
      Vectors.dense(2.0)   -> 0,
      Vectors.dense(100.0) -> 1))), (0.98 + 96.0 / 98.0) / 3.0)
  }

  /** One cluster: `b` does not exist, so the score does not either. 0.0, never the ~1.0 that
   *  standing in an infinite `b` produces. */
  test("a single cluster yields 0.0") {
    assert(silhouette(frame(Seq(
      Vectors.dense(0.0) -> 0,
      Vectors.dense(2.0) -> 0,
      Vectors.dense(9.0) -> 0))) == 0.0)
  }

  test("noise points are excluded") {
    assertClose(silhouette(frame(Seq(
      Vectors.dense(0.0)    -> 0,
      Vectors.dense(2.0)    -> 0,
      Vectors.dense(10.0)   -> 1,
      Vectors.dense(12.0)   -> 1,
      Vectors.dense(500.0)  -> -1,
      Vectors.dense(-500.0) -> -1))), 79.0 / 99.0)
  }

  /** Weighting == duplication, end to end: neighbour masses inside `a` and `b`, the cluster
   *  size that divides them, and the weighted mean over rows. */
  test("weighting equals duplication") {
    val rows = Seq(
      (Vectors.dense(0.0): Vector,  0, 2.0),
      (Vectors.dense(2.0): Vector,  0, 1.0),
      (Vectors.dense(10.0): Vector, 1, 1.0),
      (Vectors.dense(12.0): Vector, 1, 2.0))

    val duplicated = frame(rows.flatMap { case (v, l, w) => Seq.fill(w.toInt)((v, l)) })
    assertClose(silhouette(weightedFrame(rows)), silhouette(duplicated))
  }

  /** `d(p,p)` is subtracted, not assumed to be 0: under cosine a zero vector is at distance 1.0
   *  from everything INCLUDING itself. Two zero vectors in one cluster, two unit axes in the
   *  other: every a and every b is 1, so the score is exactly 0. Assuming a zero self-distance
   *  would inflate `a` to 2 and drag the score to -0.25. */
  test("cosine: a zero vector's non-zero self-distance is removed from a") {
    assertClose(silhouette(frame(Seq(
      Vectors.dense(0.0, 0.0) -> 0,
      Vectors.dense(0.0, 0.0) -> 0,
      Vectors.dense(1.0, 0.0) -> 1,
      Vectors.dense(0.0, 1.0) -> 1)), CosineDistance), 0.0)
  }
}
