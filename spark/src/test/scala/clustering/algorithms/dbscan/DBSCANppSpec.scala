package clustering.algorithms.dbscan

import clustering.benchmark.config.AlgorithmSpec
import clustering.benchmark.registry.AlgorithmRegistry
import clustering.core.Columns
import clustering.distance.EuclideanDistance
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

/** Validates `dbscanpp` against a textbook DBSCAN written from the definition.
 *
 *  This is the whole point of the exactness ladder: at `coreSampleFraction = 1.0`
 *  DBSCAN++ IS classic DBSCAN, so the distributed implementation must produce the same
 *  partition as a straight reading of the definition. Without that check, "exact" would be
 *  an unverified claim.
 *
 *  The reference is deliberately dumb — plain Scala, O(n²), no Spark — so it can be read
 *  against the definition line by line. It replaces the retired Spark-only
 *  cartesian/GraphFrames implementation, which needed a distributed graph library to say
 *  the same thing.
 */
class DBSCANppSpec extends AnyFunSuite with BeforeAndAfterAll {

  private var spark: SparkSession = _

  private val Eps    = 0.5
  private val MinPts = 4

  override def beforeAll(): Unit = {
    spark = SparkSession.builder()
      .master("local[2]")
      .appName("dbscanpp-spec")
      .config("spark.ui.enabled", "false")
      .config("spark.sql.shuffle.partitions", "4")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
  }

  /** Classic DBSCAN, straight from the definition, on the driver.
   *
   *  Core point: at least `minPts` points within ε, itself included. Two core points are in
   *  the same cluster when a chain of ε-steps between core points joins them. Every other
   *  point joins a cluster if some core point is within ε of it, else it is noise (-1).
   *  Returns one label per input point.
   */
  private def referenceDBSCAN(points: Seq[Vector], eps: Double, minPts: Int): Array[Int] = {
    val n    = points.length
    val d    = (a: Int, b: Int) => EuclideanDistance.compute(points(a), points(b))
    val core = Array.tabulate(n)(i => (0 until n).count(j => d(i, j) <= eps) >= minPts)

    // Components of the ε-graph over core points, by repeated label relaxation — slow and
    // obviously correct, which is the point of a reference implementation.
    val comp = Array.tabulate(n)(identity)
    var changed = true
    while (changed) {
      changed = false
      for (i <- 0 until n if core(i); j <- 0 until n if core(j) && d(i, j) <= eps) {
        val lo = math.min(comp(i), comp(j))
        if (comp(i) != lo || comp(j) != lo) { comp(i) = lo; comp(j) = lo; changed = true }
      }
    }

    val labels = Array.fill(n)(-1)
    for (i <- 0 until n) {
      if (core(i)) labels(i) = comp(i)
      else {
        val nearestCore = (0 until n).filter(core).sortBy(j => (d(i, j), j)).headOption
        nearestCore.foreach(j => if (d(i, j) <= eps) labels(i) = comp(j))
      }
    }
    labels
  }

  /** Same shape as [[partitionOf]], for the reference labels. */
  private def partitionOfReference(points: Seq[Vector], labels: Array[Int]): (Set[Set[String]], Set[String]) = {
    val rows  = points.map(_.toString).zip(labels)
    val noise = rows.filter(_._2 == -1).map(_._1).toSet
    val clusters = rows.filter(_._2 != -1).groupBy(_._2).values.map(_.map(_._1).toSet).toSet
    (clusters, noise)
  }

  override def afterAll(): Unit = if (spark != null) spark.stop()

  /** Two 5×5 grids of spacing 0.3 (each a single ε-connected dense blob) plus one far
   *  outlier. Deterministic by construction — no RNG, so the comparison cannot flake. */
  private val fixture: Seq[Vector] = {
    val grid = for {
      (ox, oy) <- Seq((0.0, 0.0), (10.0, 10.0))
      i        <- 0 until 5
      j        <- 0 until 5
    } yield Vectors.dense(ox + i * 0.3, oy + j * 0.3)
    grid :+ Vectors.dense(50.0, 50.0)
  }

  private def dataset(): DataFrame = {
    val session = spark
    import session.implicits._
    fixture.map(Tuple1.apply).toDF(Columns.Features)
  }

  /** The partition induced by a prediction column, as a set of point groups, plus the
   *  noise group separately. Cluster IDS are not compared — only the grouping, since ids
   *  are arbitrary between two implementations. */
  private def partitionOf(labelled: DataFrame): (Set[Set[String]], Set[String]) = {
    val rows = labelled.select(Columns.Features, Columns.Prediction).collect()
      .map(r => r.getAs[Vector](Columns.Features).toString -> r.getInt(1))
    val noise    = rows.filter(_._2 == -1).map(_._1).toSet
    val clusters = rows.filter(_._2 != -1).groupBy(_._2).values.map(_.map(_._1).toSet).toSet
    (clusters, noise)
  }

  test("coreSampleFraction = 1.0 reproduces textbook DBSCAN") {
    val data = dataset().cache()

    val (refClusters, refNoise) =
      partitionOfReference(fixture, referenceDBSCAN(fixture, Eps, MinPts))
    val pp = new DBSCANpp(eps = Eps, minPts = MinPts, coreSampleFraction = 1.0,
      distanceMetric = EuclideanDistance).fit(data).assignClusters(data)

    val (ppClusters, ppNoise) = partitionOf(pp)

    assert(refClusters.size == 2, s"fixture broken: reference found ${refClusters.size} clusters")
    assert(ppClusters == refClusters, "exact DBSCAN++ must reproduce the textbook partition")
    assert(ppNoise == refNoise, s"noise sets differ: $ppNoise vs $refNoise")
    assert(ppNoise.size == 1, "the far outlier must be the only noise point")
  }

  /** The invariant sampling must preserve, at every s: a cluster never spans two blobs.
   *
   *  The converse is NOT asserted, on purpose. Sub-sampling core candidates thins the
   *  ε-graph, so gaps wider than ε can open inside one true blob and SPLIT it — that is
   *  DBSCAN++ behaving as published (the guarantee is asymptotic in m, not per-run), and it
   *  is exactly the accuracy loss the s-sweep is supposed to measure. Pinning "exactly 2
   *  clusters" would only assert that one seed got lucky.
   */
  test("sampled DBSCAN++ never merges the two blobs, and keeps the outlier noisy") {
    val data = dataset().cache()
    Seq(0.3, 0.5, 0.8).foreach { s =>
      val model = new DBSCANpp(eps = Eps, minPts = MinPts, coreSampleFraction = s,
        distanceMetric = EuclideanDistance, seed = 7L).fit(data)
      val (clusters, noise) = partitionOf(model.assignClusters(data))

      assert(clusters.nonEmpty, s"s=$s produced no cluster at all")
      clusters.foreach { c =>
        val blobs = c.map(p => p.stripPrefix("[").split(",")(0).toDouble < 5.0)
        assert(blobs.size == 1, s"s=$s produced a cluster spanning both blobs: $c")
      }
      assert(noise.contains(Vectors.dense(50.0, 50.0).toString),
        s"s=$s lost the far outlier from the noise set")
    }
  }

  test("s = 1.0 finds exactly the two blobs — sampling is the only source of fragmentation") {
    val data  = dataset().cache()
    val model = new DBSCANpp(eps = Eps, minPts = MinPts, coreSampleFraction = 1.0,
      distanceMetric = EuclideanDistance).fit(data)
    assert(model.numClusters == 2)
  }

  test("assign: closest reproduces the paper — no noise label at all") {
    val data = dataset().cache()
    val model = new DBSCANpp(eps = Eps, minPts = MinPts, coreSampleFraction = 1.0,
      requireWithinEps = false, distanceMetric = EuclideanDistance).fit(data)
    val (_, noise) = partitionOf(model.assignClusters(data))

    assert(noise.isEmpty, "without the ε condition every point must get a cluster")
  }

  test("all three sampling strategies produce a usable clustering") {
    val data = dataset().cache()
    Seq("uniform", "linspace", "kcenter").foreach { name =>
      val model = new DBSCANpp(eps = Eps, minPts = MinPts, coreSampleFraction = 0.6,
        samplingStrategy = CandidateSelectionStrategy.fromName(name, poolFactor = 3),
        distanceMetric = EuclideanDistance, seed = 3L).fit(data)
      assert(model.numClusters >= 1, s"sampling '$name' produced no cluster")
      assert(model.corePoints.nonEmpty, s"sampling '$name' produced no core point")
    }
  }

  test("parameters that admit no core point yield an all-noise model, not a crash") {
    val data  = dataset().cache()
    val model = new DBSCANpp(eps = 0.01, minPts = 10, coreSampleFraction = 1.0,
      distanceMetric = EuclideanDistance).fit(data)
    assert(model.numClusters == 0)
    val (clusters, noise) = partitionOf(model.assignClusters(data))
    assert(clusters.isEmpty && noise.size == 51)
  }

  test("the 'dbscanexact' registry alias is the s = 1.0 code path") {
    import org.json4s.JsonDSL._
    val data = dataset().cache()

    val viaAlias = AlgorithmRegistry
      .create(AlgorithmSpec("dbscanexact",
        ("eps" -> Eps) ~ ("minPts" -> MinPts) ~ ("distance" -> "euclidean")))
      .clusterer.fit(data)
    val direct = new DBSCANpp(eps = Eps, minPts = MinPts, coreSampleFraction = 1.0,
      distanceMetric = EuclideanDistance).fit(data)

    assert(partitionOf(viaAlias.assignClusters(data)) == partitionOf(direct.assignClusters(data)))
  }

  test("'dbscanexact' rejects a sampled coreSampleFraction instead of silently ignoring it") {
    import org.json4s.JsonDSL._
    val thrown = intercept[IllegalArgumentException] {
      AlgorithmRegistry.create(AlgorithmSpec("dbscanexact",
        ("eps" -> Eps) ~ ("minPts" -> MinPts) ~ ("distance" -> "euclidean") ~
          ("coreSampleFraction" -> 0.3)))
    }
    assert(thrown.getMessage.contains("exact by definition"))
  }

  test("'dbscanexact' rejects candidate sampling params instead of silently dropping them") {
    import org.json4s.JsonDSL._
    Seq[(String, org.json4s.JObject)]("sampling" -> ("sampling" -> "kcenter"),
                                      "poolFactor" -> ("poolFactor" -> 3)).foreach {
      case (key, extra) =>
        val thrown = intercept[IllegalArgumentException] {
          AlgorithmRegistry.create(AlgorithmSpec("dbscanexact",
            ("eps" -> Eps) ~ ("minPts" -> MinPts) ~ ("distance" -> "euclidean") ~ extra))
        }
        assert(thrown.getMessage.contains(key))
    }
  }

  test("cluster ids are reproducible across repeated fits") {
    val data = dataset().cache()
    def labels() = new DBSCANpp(eps = Eps, minPts = MinPts, coreSampleFraction = 1.0,
      distanceMetric = EuclideanDistance).fit(data).coreClusterLabels.toSeq
    assert(labels() == labels(), "min-index component labelling must be deterministic")
  }
}
