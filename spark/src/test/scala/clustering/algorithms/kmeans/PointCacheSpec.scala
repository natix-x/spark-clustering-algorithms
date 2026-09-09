package clustering.algorithms.kmeans

import clustering.core.{Columns, EuclideanGeometry, SphericalGeometry, Weights}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.storage.StorageLevel
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

/** Who owns the cache of the points an algorithm iterates over.
 *
 *  The rule ([[clustering.core.Clusterer.fit]]): the CALLER caches, an algorithm caches only
 *  frames it creates itself. [[clustering.benchmark.SparkClusteringJob]] persists the loaded
 *  data, and `Weights.withWeights` plus a no-op `Geometry.prepare` are bare projections of it —
 *  persisting those stores a second copy of `features`, which on Gaia (~26 GB) or Cohere
 *  (~196 GB) is what pushes both copies to disk and turns every later pass into a disk read.
 *
 *  Checked here because the failure is invisible in results: labels stay correct, the run is just
 *  slower and spills more. `LloydKMeans` is the seam where the decision lives, so the k-means
 *  family (plain, breathing, bisecting) is covered by testing it directly.
 */
class PointCacheSpec extends AnyFunSuite with BeforeAndAfterAll {

  private var spark: SparkSession = _

  override def beforeAll(): Unit = {
    spark = SparkSession.builder()
      .master("local[2]")
      .appName("point-cache-spec")
      .config("spark.ui.enabled", "false")
      .config("spark.sql.shuffle.partitions", "4")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
  }

  override def afterAll(): Unit = if (spark != null) spark.stop()

  private def points(n: Int): DataFrame = {
    val session = spark
    import session.implicits._
    Seq.tabulate(n)(i => Tuple1(Vectors.dense((i % 5).toDouble, (i % 7).toDouble): Vector))
      .toDF(Columns.Features)
  }

  test("withWeights leaves the projection uncached — the caller's cache is what gets read") {
    assert(Weights.withWeights(points(200)).storageLevel == StorageLevel.NONE)
  }

  test("euclidean k-means does not cache a second copy of the caller's data") {
    val data = points(200).persist(StorageLevel.MEMORY_AND_DISK)
    try {
      data.count()
      val setup = LloydKMeans.initialize(data, EuclideanGeometry, k = 2, seed = 1L,
        StorageLevel.MEMORY_AND_DISK)
      try {
        assert(setup.preparedPoints.storageLevel == StorageLevel.NONE,
          "a no-op prepare over cached data must not be cached again")
        assert(!setup.ownsCache)
      } finally setup.release()
      assert(data.storageLevel != StorageLevel.NONE, "release() dropped a cache it does not own")
    } finally data.unpersist(blocking = true)
  }

  test("spherical k-means caches the normalised copy it creates") {
    val data = points(200).persist(StorageLevel.MEMORY_AND_DISK)
    try {
      data.count()
      val setup = LloydKMeans.initialize(data, SphericalGeometry, k = 2, seed = 1L,
        StorageLevel.MEMORY_AND_DISK)
      try {
        assert(setup.preparedPoints.storageLevel != StorageLevel.NONE,
          "normalising every row is real work — recomputing it per iteration is the wrong trade")
        assert(setup.ownsCache)
      } finally setup.release()
    } finally data.unpersist(blocking = true)
  }
}
