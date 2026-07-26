package clustering.algorithms.kmeans

import clustering.core.Columns
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.SparkSession
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

/** [[LloydKMeans.sampleInitialCentroids]] must honour weights (A-Res weighted reservoir sampling),
 *  not just sample uniformly over rows — otherwise it would break the project-wide
 *  "weighting == duplication" invariant tested in `WeightedClusteringSpec`. */
class LloydKMeansSpec extends AnyFunSuite with BeforeAndAfterAll {

  private var spark: SparkSession = _

  override def beforeAll(): Unit = {
    spark = SparkSession.builder()
      .master("local[2]")
      .appName("lloyd-spec")
      .config("spark.ui.enabled", "false")
      .config("spark.sql.shuffle.partitions", "4")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
  }

  override def afterAll(): Unit = if (spark != null) spark.stop()

  test("sampleInitial: a massively heavier point dominates a size-1 sample across seeds") {
    val session = spark
    import session.implicits._
    // One heavy point (weight 1000) among nine light ones (weight 1 each): a uniform sample
    // over rows would pick the heavy point ~10% of the time; weighted sampling should pick it
    // almost always.
    val heavy = Vectors.dense(100.0)
    val df = ((heavy -> 1000.0) +: (0 until 9).map(i => Vectors.dense(i.toDouble) -> 1.0))
      .toDF(Columns.Features, Columns.Weight)

    val picks = (0 until 20).map(seed => LloydKMeans.sampleInitialCentroids(df, count = 1, seed = seed.toLong).head)
    val heavyPicks = picks.count(_ == heavy)
    assert(heavyPicks >= 18, s"expected the weight-1000 point to dominate, got $heavyPicks/20 seeds")
  }

  test("sampleInitial: unit weights reduce to plain uniform sampling (no bias toward row order)") {
    val session = spark
    import session.implicits._
    val points: Seq[Vector] = (0 until 5).map(i => Vectors.dense(i.toDouble))
    val df = points.map(v => v -> 1.0).toDF(Columns.Features, Columns.Weight)

    val distinctPicks = (0 until 30)
      .map(seed => LloydKMeans.sampleInitialCentroids(df, count = 1, seed = seed.toLong).head)
      .toSet
    assert(distinctPicks.size > 1, "expected varied picks across seeds under unit weights")
  }
}