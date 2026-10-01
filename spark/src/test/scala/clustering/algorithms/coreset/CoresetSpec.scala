package clustering.algorithms.coreset

import clustering.algorithms.coreset.components.LightweightCoreset
import clustering.algorithms.coreset.inner.{LocalLloyd, LocalMedoids}
import clustering.benchmark.config.AlgorithmSpec
import clustering.benchmark.registry.AlgorithmRegistry
import clustering.core.{Columns, EuclideanGeometry, NearestPrototypeModel, SphericalGeometry, Weights}
import clustering.distance.EuclideanDistance
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.storage.StorageLevel
import org.json4s.JsonDSL._
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

import scala.util.Random

/** Slot 8 — the lightweight coreset (KDD 2018) and the three inner algorithms that compose with
 *  it.
 *
 *  What is actually asserted, and why each assertion is the one that can fail silently:
 *
 *   - **the reduction is unbiased**, not merely "small". A coreset whose weights do not reproduce
 *     the data's mass, or whose cost estimate is off, still clusters into something plausible —
 *     every downstream number would be wrong by an amount no cluster count reveals. So the mass
 *     and the cost of a FIXED solution are compared against the full data directly.
 *   - **weighting == duplication** for the inner solvers, which is the invariant that lets the
 *     coreset stand in for the data at all.
 *
 *  The `hac` inner and its own assertions (the nearest-neighbour chain against a stepwise
 *  agglomeration written from the definition) went with the inner itself on 14.09.2026 — reason in
 *  `algorithm_selection.md` §8.
 */
class CoresetSpec extends AnyFunSuite with BeforeAndAfterAll {

  private var spark: SparkSession = _

  override def beforeAll(): Unit = {
    spark = SparkSession.builder()
      .master("local[2]")
      .appName("coreset-spec")
      .config("spark.ui.enabled", "false")
      .config("spark.sql.shuffle.partitions", "4")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
  }

  override def afterAll(): Unit = if (spark != null) spark.stop()

  private val blobCentres = Array(Array(0.0, 0.0), Array(40.0, 0.0), Array(0.0, 40.0))

  /** Three well-separated Gaussian blobs, 400 rows each. Separated so that "recovered the
   *  clusters" is a statement about the reduction, not about the inner solver's luck. */
  private def blobs(seed: Long = 7L): Seq[(Vector, Int)] = {
    val rng = new Random(seed)
    for {
      (centre, label) <- blobCentres.toSeq.zipWithIndex
      _ <- 0 until 400
    } yield (Vectors.dense(centre(0) + rng.nextGaussian() * 1.5, centre(1) + rng.nextGaussian() * 1.5), label)
  }

  private def frameOf(points: Seq[Vector]): DataFrame =
    spark.createDataFrame(points.map(Tuple1(_))).toDF(Columns.Features)

  private def blobFrame(seed: Long = 7L): DataFrame = frameOf(blobs(seed).map(_._1))

  /** sum_x w_x * min_c d(x, c)^2 over the whole frame — the quantity a coreset must preserve. */
  private def cost(data: DataFrame, centres: Array[Array[Double]]): Double =
    Weights.toRdd(Weights.withWeights(data)).map { case (features, weight) =>
      val coords = features.toArray
      val nearest = centres.map(EuclideanDistance.compute(coords, _)).min
      weight * nearest * nearest
    }.sum()

  private def localCost(
    points: Array[Array[Double]], weights: Array[Double], centres: Array[Array[Double]]
  ): Double = {
    var total = 0.0
    var j = 0
    while (j < points.length) {
      val nearest = centres.map(EuclideanDistance.compute(points(j), _)).min
      total += weights(j) * nearest * nearest
      j += 1
    }
    total
  }

  // --- the reduction ---------------------------------------------------------------------------

  test("the draw reproduces the data's mass and never emits a non-positive weight") {
    val data = blobFrame().persist()
    val prepared = Weights.withWeights(data)

    // Several seeds: one draw's mass is a random variable, and the claim is about the estimator.
    val ratios = (1 to 5).map { seed =>
      val coreset = LightweightCoreset.build(prepared, m = 300, EuclideanDistance, seed = seed)
      assert(coreset.size > 0)
      assert(coreset.weights.forall(_ > 0.0), "a coreset row with weight <= 0 would vanish silently")
      assert(coreset.points.forall(_.length == 2))
      // Poisson draw: the size is m in expectation, not exactly m.
      assert(math.abs(coreset.size - 300) < 120, s"drew ${coreset.size} rows for m=300")
      coreset.coresetMass / coreset.mass
    }

    val meanRatio = ratios.sum / ratios.length
    assert(math.abs(meanRatio - 1.0) < 0.05,
      s"coreset mass should equal the data's in expectation, got ratio $meanRatio (per seed: $ratios)")
    data.unpersist()
  }

  test("the coreset preserves the cost of a fixed solution — the guarantee that makes it usable") {
    val data = blobFrame().persist()
    val prepared = Weights.withWeights(data)
    val centres = blobCentres.map(_.clone())
    val exact = cost(data, centres)

    val errors = (1 to 5).map { seed =>
      val coreset = LightweightCoreset.build(prepared, m = 400, EuclideanDistance, seed = seed)
      val estimate = localCost(coreset.points, coreset.weights, centres)
      math.abs(estimate - exact) / exact
    }

    assert(errors.sum / errors.length < 0.15,
      s"mean relative cost error ${errors.sum / errors.length} — the reduction is biased or mis-weighted")
    data.unpersist()
  }

  test("importance sampling keeps outliers a uniform draw would miss") {
    // One far outlier among 2000 dense rows: a uniform draw takes it with probability m/n, the
    // lightweight coreset with probability driven by its distance to the mean, which dominates the
    // dispersion here. This is the whole reason the construction is not uniform sampling.
    val rng = new Random(3L)
    val dense = (0 until 2000).map(_ => Vectors.dense(rng.nextGaussian(), rng.nextGaussian()))
    val outlier = Vectors.dense(500.0, 500.0)
    val data = frameOf(dense :+ outlier).persist()
    val prepared = Weights.withWeights(data)

    val hits = (1 to 10).count { seed =>
      val coreset = LightweightCoreset.build(prepared, m = 50, EuclideanDistance, seed = seed)
      coreset.points.exists(p => p(0) > 100.0)
    }
    assert(hits == 10, s"the outlier was drawn in only $hits/10 draws; q(x) should make it certain")
    data.unpersist()
  }

  // --- inner: kmeans / medoids -----------------------------------------------------------------

  test("every inner algorithm recovers three separated blobs through the reduction") {
    val labelled = blobs()
    val data = frameOf(labelled.map(_._1)).persist()

    // Lloyd's init is a plain weighted random draw here, exactly as it is in the distributed
    // `kmeans` entry, so it can land two centroids in one blob and converge to a local optimum —
    // a property of the algorithm, not of the reduction. The seed is therefore fixed, and the
    // assertion below is about the coreset carrying the structure, not about the init's luck.
    val solvers = Seq(
      new LocalLloyd(k = 3, maxIter = 50, eps = 1e-4, geometry = EuclideanGeometry, seed = 1L),
      new LocalMedoids("fastpam", k = 3, maxIter = 50, distance = EuclideanDistance, seed = 11L),
      new LocalMedoids("fasterpam", k = 3, maxIter = 50, distance = EuclideanDistance, seed = 11L)
    )

    solvers.foreach { solver =>
      val model = new Coreset(k = 3, m = 200, solver = solver, seed = 5L).fit(data)
      assert(model.prototypeVectors.length == 3, s"${solver.name} produced ${model.prototypeVectors.length} prototypes")

      val predicted = model.assignClusters(data).select(Columns.Prediction).collect().map(_.getInt(0))
      val perTrueBlob = labelled.map(_._2).zip(predicted).groupBy(_._1).map { case (blob, pairs) =>
        blob -> pairs.map(_._2).distinct
      }
      perTrueBlob.foreach { case (blob, labels) =>
        assert(labels.length == 1, s"${solver.name}: true blob $blob split across labels $labels")
      }
      assert(perTrueBlob.values.flatten.toSet.size == 3,
        s"${solver.name}: the three blobs collapsed into ${perTrueBlob.values.flatten.toSet}")
    }
    data.unpersist()
  }

  test("inner: medoids returns real coreset rows, as the medoid objective requires") {
    val data = blobFrame().persist()
    val model = new Coreset(k = 3, m = 150,
      solver = new LocalMedoids("fastpam", 3, 50, EuclideanDistance, 11L), seed = 9L).fit(data)
    val rows = data.collect().map(_.getAs[Vector](Columns.Features).toArray)
    model.prototypeVectors.map(_.toArray).foreach { medoid =>
      assert(rows.exists(row => row.sameElements(medoid)),
        s"medoid ${medoid.mkString(",")} is not a data point")
    }
    data.unpersist()
  }

  test("weighting == duplication for the inner solvers") {
    val groups = Seq(
      Array(0.0, 0.0) -> 3.0,
      Array(0.5, 0.2) -> 1.0,
      Array(10.0, 10.0) -> 2.0,
      Array(10.4, 9.7) -> 4.0,
      Array(20.0, 0.0) -> 1.0
    )
    val weighted = groups.map(_._1).toArray
    val weights = groups.map(_._2).toArray
    val duplicated = groups.flatMap { case (row, w) => Seq.fill(w.toInt)(row) }.toArray
    val unit = Weights.unit(duplicated.length)

    // Lloyd: same centroids, from an init pinned to the same three points. Without pinning the
    // two forms draw different starting centroids — the init is a weighted reservoir sample — and
    // would be compared across two different local optima, which says nothing about weighting.
    val lloyd = new LocalLloyd(k = 3, maxIter = 100, eps = 1e-9, geometry = EuclideanGeometry, seed = 4L)
    val init = Array(Array(0.0, 0.0), Array(10.0, 10.0), Array(20.0, 0.0))
    val weightedCentres = lloyd.iterate(weighted, weights, init).prototypes.map(_.toSeq).sortBy(_.head)
    val duplicatedCentres = lloyd.iterate(duplicated, unit, init).prototypes.map(_.toSeq).sortBy(_.head)
    weightedCentres.zip(duplicatedCentres).foreach { case (a, b) =>
      a.zip(b).foreach { case (x, y) => assert(math.abs(x - y) < 1e-9, s"$a vs $b") }
    }
  }


  test("a model fitted on a coreset labels by nearest prototype") {
    val data = blobFrame().persist()
    val model = new Coreset(k = 3, m = 200,
      solver = new LocalLloyd(3, 50, 1e-4, EuclideanGeometry, 11L), seed = 2L).fit(data)
    val prototypes = model.prototypeVectors.map(_.toArray)
    val rows = model.assignClusters(data).collect()
    rows.foreach { row =>
      val coords = row.getAs[Vector](Columns.Features).toArray
      assert(row.getAs[Int](Columns.Prediction) ==
        NearestPrototypeModel.nearestRaw(coords, prototypes, EuclideanDistance))
    }
    assert(model.coresetSize > 0 && model.requestedSize == 200 && model.inner == "kmeans")
    // What reaches the result contract is the REALISED draw, not the requested m: the run's
    // quality belongs to the rows it actually clustered.
    assert(model.reductionSize.contains(model.coresetSize))
    data.unpersist()
  }

  test("the construction does not cache a second copy of the caller's data") {
    // The rule from `Clusterer.fit`: the caller caches, an algorithm caches only frames it
    // creates. The construction reads the input three times, which is exactly the situation where
    // an extra `persist` looks harmless and silently doubles a 196 GB dataset on disk.
    val data = blobFrame().persist(StorageLevel.MEMORY_AND_DISK)
    try {
      data.count()
      val cachedBefore = spark.sparkContext.getPersistentRDDs.size

      new Coreset(3, 200, new LocalLloyd(3, 20, 1e-4, EuclideanGeometry, 1L), 5L).fit(data)
      assert(spark.sparkContext.getPersistentRDDs.size == cachedBefore,
        "a no-op prepare over cached data must not be cached again")

      // Spherical DOES create a frame of its own (it normalises every row), and must release it.
      new Coreset(3, 200, new LocalLloyd(3, 20, 1e-4, SphericalGeometry, 1L), 5L).fit(data)
      assert(spark.sparkContext.getPersistentRDDs.size == cachedBefore,
        "the normalised copy this fit created was not released")
      assert(data.storageLevel != StorageLevel.NONE, "the fit dropped a cache it does not own")
    } finally data.unpersist(blocking = true)
  }

  // --- the registry ----------------------------------------------------------------------------

  test("the registry builds every inner mode and rejects unknown ones") {
    val base = ("k" -> 3) ~ ("m" -> 100) ~ ("distance" -> "euclidean")
    Seq("kmeans", "medoids").foreach { inner =>
      val built = AlgorithmRegistry.create(AlgorithmSpec("coreset", base ~ ("inner" -> inner)))
      assert(built.clusterer.isInstanceOf[Coreset], s"inner=$inner did not build a Coreset")
    }

    assert(intercept[IllegalArgumentException] {
      AlgorithmRegistry.create(AlgorithmSpec("coreset", base ~ ("inner" -> "birch")))
    }.getMessage.contains("Unknown inner algorithm"))

    // `m` has no default on purpose: a result that hides its coreset size cannot enter the sweep.
    assert(intercept[IllegalArgumentException] {
      AlgorithmRegistry.create(AlgorithmSpec("coreset", ("k" -> 3) ~ ("distance" -> "euclidean")))
    }.getMessage.contains("'m'"))
  }

  test("spherical geometry is threaded through the reduction") {
    // Directions on a circle: normalised data, so the coreset's distances and the inner solve both
    // live on the unit sphere. Only the pipeline wiring is asserted — that the fit runs and its
    // prototypes are unit vectors, which they cannot be unless `prepare` ran before the draw.
    val rng = new Random(13L)
    val rows = (0 until 300).map { i =>
      val angle = if (i % 2 == 0) 0.2 + rng.nextGaussian() * 0.05 else 2.6 + rng.nextGaussian() * 0.05
      val radius = 1.0 + rng.nextDouble() * 9.0
      Vectors.dense(radius * math.cos(angle), radius * math.sin(angle))
    }
    val data = frameOf(rows).persist()
    val built = AlgorithmRegistry.create(AlgorithmSpec("coreset",
      ("k" -> 2) ~ ("m" -> 150) ~ ("inner" -> "kmeans") ~ ("geometry" -> "spherical")))
    val model = built.clusterer.fit(data).asInstanceOf[CoresetModel]
    model.prototypeVectors.foreach { centroid =>
      val norm = math.sqrt(centroid.toArray.map(x => x * x).sum)
      assert(math.abs(norm - 1.0) < 1e-9, s"spherical centroid is not on the unit sphere: norm=$norm")
    }
    data.unpersist()
  }
}
