package clustering.algorithms.kmeans

import clustering.core.{Columns, Weights}
import clustering.distance.EuclideanDistance
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

import scala.util.Random

/** `refine: breathing` (Fritzke 2020–2023) against `refine: none` on a fixture built to defeat
 *  plain Lloyd.
 *
 *  The point of the knob is escaping a local minimum that no assignment step can fix: five tight,
 *  far-apart groups with k = 5, but a seed whose uniform initial sample puts two centroids in one
 *  group and none in another. Lloyd converges there and stays; breathing inserts a centroid where
 *  the error is largest, then deletes the least useful one, which is exactly the move Lloyd
 *  cannot make.
 */
class BreathingKMeansSpec extends AnyFunSuite with BeforeAndAfterAll {

  private var spark: SparkSession = _

  override def beforeAll(): Unit = {
    spark = SparkSession.builder()
      .master("local[2]")
      .appName("breathing-kmeans-spec")
      .config("spark.ui.enabled", "false")
      .config("spark.sql.shuffle.partitions", "4")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
  }

  override def afterAll(): Unit = if (spark != null) spark.stop()

  private val centres = Seq((0.0, 0.0), (100.0, 0.0), (0.0, 100.0), (100.0, 100.0), (50.0, 200.0))

  private lazy val data: DataFrame = {
    val session = spark
    import session.implicits._
    val rnd = new Random(3L)
    centres.flatMap { case (cx, cy) =>
      (0 until 120).map(_ => Vectors.dense(cx + rnd.nextGaussian() * 1.5, cy + rnd.nextGaussian() * 1.5))
    }.map(Tuple1.apply).toDF(Columns.Features).repartition(4).cache()
  }

  /** φ(C, X) = Σ w · d(x, nearest)², computed independently of the algorithms under test. */
  private def sse(centroids: Array[Vector]): Double =
    data.collect().map(_.getAs[Vector](Columns.Features)).map { x =>
      val d = centroids.map(c => EuclideanDistance.compute(x, c)).min
      d * d
    }.sum

  private def plain(seed: Long)      = new KMeans(k = 5, maxIter = 50, seed = seed).fit(data).centroids
  private def breathing(seed: Long)  = new BreathingKMeans(k = 5, m0 = 3, maxIter = 50,
    seed = seed).fit(data).centroids

  test("breathing returns exactly k centroids") {
    assert(breathing(11L).length == 5)
  }

  test("breathing never scores worse than plain k-means, and escapes a bad start") {
    // Seeds are scanned so the test asserts the real claim (breathing ≥ plain everywhere, and
    // strictly better where Lloyd gets stuck) instead of pinning one lucky seed.
    val results = Seq(1L, 7L, 11L, 23L).map { s =>
      val p = sse(plain(s))
      val b = sse(breathing(s))
      assert(b <= p * 1.000001, s"seed $s: breathing SSE $b worse than plain $p")
      (s, p, b)
    }
    assert(results.exists { case (_, p, b) => b < p * 0.99 },
      s"expected at least one seed where breathing improves clearly, got $results")
  }

  test("breathing recovers one centroid per group where plain k-means may not") {
    val cs   = breathing(11L)
    val hits = centres.map { case (cx, cy) =>
      cs.count(c => EuclideanDistance.compute(c, Vectors.dense(cx, cy)) < 10.0)
    }
    assert(hits.forall(_ == 1), s"expected one centroid per group, got $hits")
  }

  test("breathing honours weights: the error it minimises is the weighted one") {
    val session = spark
    import session.implicits._
    // Two groups, the right one 9× heavier. With k = 1 the single centroid must land on the
    // weighted mean, i.e. near the heavy side.
    val df = Seq((Vectors.dense(0.0), 1.0), (Vectors.dense(10.0), 9.0))
      .toDF(Columns.Features, Columns.Weight)
    val c = new BreathingKMeans(k = 1, m0 = 2, maxIter = 20, seed = 5L)
      .fit(df).centroids.head
    assert(math.abs(c(0) - 9.0) < 1e-9, s"expected weighted mean 9.0, got ${c(0)}")
    assert(Weights.isWeighted(df))
  }

  test("an unknown refine mode is rejected by the registry") {
    import clustering.benchmark.config.AlgorithmSpec
    import clustering.benchmark.registry.AlgorithmRegistry
    import org.json4s.JsonDSL._
    val thrown = intercept[IllegalArgumentException] {
      AlgorithmRegistry.create(AlgorithmSpec("kmeans",
        ("k" -> 3) ~ ("distance" -> "euclidean") ~ ("refine" -> "hyperventilating")))
    }
    assert(thrown.getMessage.contains("Unknown refine mode"))
  }
}
