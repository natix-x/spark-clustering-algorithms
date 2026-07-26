package clustering.core

import clustering.algorithms.density.DBSCANpp
import clustering.algorithms.kmeans.{BisectingKMeans, KMeans}
import clustering.algorithms.kmedoids.{FastPAM, FasterPAM, PAM}
import clustering.distance.EuclideanDistance
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

/** Weights are a `core` feature, so the invariant is checked once, for every clusterer:
 *  **weighting is duplication.** Clustering a row of weight w must equal clustering w copies
 *  of that row.
 *
 *  This is the property that makes any data-reduction technique (slot 8's coresets, a
 *  pre-aggregated grid, a stratified sample) compose with any algorithm — without it a
 *  reduction silently changes the objective being optimised.
 */
class WeightedClusteringSpec extends AnyFunSuite with BeforeAndAfterAll {

  private var spark: SparkSession = _

  override def beforeAll(): Unit = {
    spark = SparkSession.builder()
      .master("local[2]")
      .appName("weighted-clustering-spec")
      .config("spark.ui.enabled", "false")
      .config("spark.sql.shuffle.partitions", "4")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
  }

  override def afterAll(): Unit = if (spark != null) spark.stop()

  /** Three tight groups; the middle one is deliberately over-represented so that ignoring
   *  weights visibly moves the answer. */
  private val groups: Seq[(Vector, Int)] = Seq(
    Vectors.dense(0.0, 0.0)   -> 1,
    Vectors.dense(0.4, 0.0)   -> 1,
    Vectors.dense(10.0, 0.0)  -> 5,
    Vectors.dense(10.4, 0.0)  -> 5,
    Vectors.dense(30.0, 0.0)  -> 2,
    Vectors.dense(30.4, 0.0)  -> 2
  )

  private def weighted: DataFrame = {
    val session = spark
    import session.implicits._
    groups.map { case (v, w) => (v, w.toDouble) }
      .toDF(Columns.Features, Columns.Weight)
      .repartition(3)
      .cache()
  }

  private def duplicated: DataFrame = {
    val session = spark
    import session.implicits._
    groups.flatMap { case (v, w) => Seq.fill(w)(v) }
      .map(Tuple1.apply)
      .toDF(Columns.Features)
      .repartition(3)
      .cache()
  }

  private def show(vs: Array[Vector]): String = vs.map(_.toString).sorted.mkString(", ")

  test("Weights: absent column means unit weights, present column is honoured") {
    assert(!Weights.isWeighted(duplicated))
    assert(Weights.isWeighted(weighted))
    assert(Weights.withWeights(duplicated).columns.toSeq == Seq(Columns.Features, Columns.Weight))
    assert(Weights.toRdd(duplicated).map(_._2).distinct().collect().toSeq == Seq(1.0))
    assert(Weights.toRdd(weighted).map(_._2).sum() == groups.map(_._2).sum.toDouble)
  }

  test("kmeans: weighted centroids equal centroids of the duplicated data") {
    def run(df: DataFrame) = new KMeans(k = 3, maxIter = 20, seed = 1L)
      .fit(df).centroids
    val w = run(weighted)
    val d = run(duplicated)
    // Same k, same geometry, deterministic init on identical point sets ⇒ same centroids.
    assert(show(w) == show(d), s"weighted=${show(w)} duplicated=${show(d)}")
  }

  test("kmeans: ignoring weights would move the centroid — the weighted mean really differs") {
    val session = spark
    import session.implicits._
    // One cluster only: its centroid is the weighted mean, which must sit near the heavy point.
    val df = Seq((Vectors.dense(0.0), 1.0), (Vectors.dense(10.0), 9.0))
      .toDF(Columns.Features, Columns.Weight)
    val centroid = new KMeans(k = 1, maxIter = 5, seed = 1L)
      .fit(df).centroids.head
    assert(math.abs(centroid(0) - 9.0) < 1e-9, s"expected weighted mean 9.0, got ${centroid(0)}")
  }

  test("pam / fastpam / fasterpam: weighted medoids equal medoids of the duplicated data") {
    val pts  = groups.map(_._1).toArray
    val ws   = groups.map(_._2.toDouble).toArray
    val dup  = groups.flatMap { case (v, w) => Seq.fill(w)(v) }.toArray

    Seq[(String, Array[Vector], Array[Double]) => Array[Vector]](
      (_, p, w) => new PAM(3, 50, EuclideanDistance).fitLocal(p, w).medoids,
      (_, p, w) => new FastPAM(3, 50, EuclideanDistance).fitLocal(p, w).medoids,
      (_, p, w) => new FasterPAM(3, 50, EuclideanDistance, 7L).fitLocal(p, w).medoids
    ).zip(Seq("pam", "fastpam", "fasterpam")).foreach { case (solver, name) =>
      val weightedMedoids = solver(name, pts, ws)
      val dupMedoids      = solver(name, dup, Weights.unit(dup.length))
      assert(show(weightedMedoids) == show(dupMedoids),
        s"$name: weighted=${show(weightedMedoids)} duplicated=${show(dupMedoids)}")
    }
  }

  test("bisectingkmeans: weighted run equals the duplicated run") {
    def run(df: DataFrame) = new BisectingKMeans(k = 3, maxIter = 20, seed = 2L)
      .fit(df).clusterCentroids
    assert(show(run(weighted)) == show(run(duplicated)))
  }

  test("dbscanpp: minPts is a mass threshold, so weights decide core points") {
    // Each group holds 2 rows within eps; unweighted that is a mass of 2, weighted it is 2*w.
    // minPts = 4 therefore keeps only the groups whose weight is at least 2.
    val model = new DBSCANpp(eps = 1.0, minPts = 4, coreSampleFraction = 1.0,
      distance = EuclideanDistance).fit(weighted)
    val coreXs = model.cores.map(_(0)).sorted.toSeq
    assert(coreXs == Seq(10.0, 10.4, 30.0, 30.4),
      s"expected the weight-5 and weight-2 groups to be core, got $coreXs")

    // The same data duplicated must give the same cores — weighting is duplication.
    val dupModel = new DBSCANpp(eps = 1.0, minPts = 4, coreSampleFraction = 1.0,
      distance = EuclideanDistance).fit(duplicated)
    assert(dupModel.cores.map(_(0)).distinct.sorted.toSeq == coreXs)
    assert(dupModel.labels.distinct.length == model.labels.distinct.length)
  }
}
