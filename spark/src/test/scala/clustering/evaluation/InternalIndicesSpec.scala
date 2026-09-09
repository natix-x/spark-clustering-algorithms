package clustering.evaluation

import clustering.core.{Columns, Model}
import clustering.distance.EuclideanDistance
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

/** The two centroid-based internal indices, pinned against values computed BY HAND from the
 *  published definitions — the same role `DBSCANppSpec`'s textbook DBSCAN plays for the density
 *  slot. Both are reported per run and compared across engines, so a silent formula slip here
 *  would look like a genuine quality difference between Spark and Flink.
 *
 *  The reference labelling is fixed by the test (a `Model` that reads a label column), so what is
 *  under test is the index arithmetic and the distributed moment computation, never a clusterer.
 */
class InternalIndicesSpec extends AnyFunSuite with BeforeAndAfterAll {

  private var spark: SparkSession = _

  override def beforeAll(): Unit = {
    spark = SparkSession.builder()
      .master("local[2]")
      .appName("internal-indices-spec")
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

  private def dbi(data: DataFrame): Double =
    new DaviesBouldinEvaluator(EuclideanDistance).evaluate(LabelColumnModel, data)

  private def ch(data: DataFrame): Double =
    new CalinskiHarabaszEvaluator(EuclideanDistance).evaluate(LabelColumnModel, data)

  /** Two clusters of two points on a line: {0, 2} and {10, 12}.
   *  Centroids 1 and 11, S_0 = S_1 = 1, M_01 = 10  ->  DB = (1+1)/10 = 0.2.
   *  Grand centroid 6: between = 2·5² + 2·5² = 100, within = 4·1² = 4,
   *  k = 2, n = 4  ->  CH = (100/1) / (4/2) = 50. */
  private val twoPairs: Seq[(Vector, Int)] = Seq(
    Vectors.dense(0.0)  -> 0,
    Vectors.dense(2.0)  -> 0,
    Vectors.dense(10.0) -> 1,
    Vectors.dense(12.0) -> 1
  )

  test("Davies-Bouldin matches the hand-computed value") {
    assert(math.abs(dbi(frame(twoPairs)) - 0.2) < 1e-9)
  }

  test("Calinski-Harabasz matches the hand-computed value") {
    assert(math.abs(ch(frame(twoPairs)) - 50.0) < 1e-9)
  }

  /** Direction of each index — the property the analysis actually relies on, and the one that
   *  a sign or an inverted ratio would break while the magnitudes still look plausible. */
  test("separating the clusters lowers Davies-Bouldin and raises Calinski-Harabasz") {
    val separated = frame(twoPairs)
    val overlapping = frame(Seq(
      Vectors.dense(0.0) -> 0,
      Vectors.dense(2.0) -> 0,
      Vectors.dense(1.0) -> 1,
      Vectors.dense(3.0) -> 1
    ))
    assert(dbi(separated) < dbi(overlapping))
    assert(ch(separated) > ch(overlapping))
  }

  /** Weighting == duplication, the repo-wide invariant: both indices aggregate MASS, so a point
   *  of weight w must score exactly as w copies of it. */
  test("weighting equals duplication for both indices") {
    val weights = Seq(
      (Vectors.dense(0.0): Vector, 0, 3.0),
      (Vectors.dense(2.0): Vector, 0, 1.0),
      (Vectors.dense(10.0): Vector, 1, 1.0),
      (Vectors.dense(12.0): Vector, 1, 2.0)
    )
    val duplicated = frame(weights.flatMap { case (v, l, w) => Seq.fill(w.toInt)((v, l)) })
    val weighted   = weightedFrame(weights)

    assert(math.abs(dbi(weighted) - dbi(duplicated)) < 1e-9)
    assert(math.abs(ch(weighted) - ch(duplicated)) < 1e-9)
  }

  /** Noise (label -1) is excluded, exactly as it is from the silhouette — otherwise a DBSCAN run
   *  would be scored on a "cluster" made of everything the algorithm refused to cluster. */
  test("noise points are excluded from both indices") {
    val withNoise = frame(twoPairs ++ Seq(
      Vectors.dense(500.0)  -> -1,
      Vectors.dense(-500.0) -> -1))

    assert(math.abs(dbi(withNoise) - 0.2) < 1e-9)
    assert(math.abs(ch(withNoise) - 50.0) < 1e-9)
  }

  /** Fewer than two clusters: both indices are undefined and report 0.0, the convention the
   *  other evaluators already use for "could not be computed". */
  test("a single cluster yields 0.0 for both indices") {
    val single = frame(Seq(
      Vectors.dense(0.0) -> 0,
      Vectors.dense(2.0) -> 0))

    assert(dbi(single) == 0.0)
    assert(ch(single) == 0.0)
  }

  /** Coincident centroids: the pair contributes nothing instead of an Infinity that would not
   *  even serialise as JSON (scikit-learn's `davies_bouldin_score` does the same). */
  test("Davies-Bouldin stays finite when two clusters share a centroid") {
    val coincident = frame(Seq(
      Vectors.dense(0.0) -> 0,
      Vectors.dense(2.0) -> 0,
      Vectors.dense(0.0) -> 1,
      Vectors.dense(2.0) -> 1))

    assert(dbi(coincident) == 0.0)
  }
}
