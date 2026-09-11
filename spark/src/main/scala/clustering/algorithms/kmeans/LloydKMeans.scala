package clustering.algorithms.kmeans

import clustering.core.{Columns, Geometry, NearestPrototypeModel, Weights}
import clustering.distance.DistanceMetric
import clustering.utils.{Convergence, PartitionAggregator}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.storage.StorageLevel

/** Lloyd's iteration, extracted so that both [[KMeans]] and [[BreathingKMeans]] run the SAME
 *  loop — the latter only wraps it in an outer add/remove cycle, which is what makes
 *  `refine: breathing` a knob on one algorithm rather than a second algorithm.
 *
 *  Operates on ALREADY PREPARED data: `[features, weight]`, projected into the geometry's space
 *  (see [[clustering.core.Weights]] and [[Geometry]]). The number of clusters is
 *  `initial.length`, not a field, so the same loop serves k and k+m centroids.
 *
 *  Breathing-only statistics (mass/error/utility per centroid) live in [[BreathingKMeans]]
 *  itself, not here — this object stays the shared kernel every centroid-based algorithm
 *  (`KMeans`, `BreathingKMeans`, `BisectingKMeans`) uses, not a dumping ground for one knob's
 *  private math.
 */
private[kmeans] object LloydKMeans {

  /** The prepared points plus everything derived from them once per fit.
   *
   *  `release()` is a no-op unless this fit created the cache — the caller's cache is never
   *  unpersisted from in here (see [[clustering.core.Clusterer.fit]]). */
  final case class LloydContext(
    preparedPoints:   DataFrame,
    fitDistance: DistanceMetric,
    initialCentroids: Array[Vector],
    ownsCache: Boolean
  ) {
    def release(): Unit = if (ownsCache) preparedPoints.unpersist(blocking = false)
  }

  def initialize(
    data: DataFrame,
    geometry: Geometry,
    k: Int,
    seed: Long,
    storageLevel: StorageLevel
  ): LloydContext = {
    val fitDistance = geometry.fitDistance
    // Cached only when `prepare` actually computes something (spherical normalises every row, so
    // every iteration would redo the UDF). A no-op `prepare` leaves a bare projection of the
    // caller's data, whose cache Catalyst reuses — persisting it would just duplicate it.
    val projected = geometry.prepare(Weights.withWeights(data))
    val ownsCache = geometry.transformsFeatures
    val preparedPoints = if (ownsCache) projected.persist(storageLevel) else projected

    val initialCentroids = sampleInitialCentroids(preparedPoints, k, seed)
    // by-name message: this count() job only runs on the failure path, not on every fit.
    require(initialCentroids.length == k,
      s"Could not sample $k initialCentroids centroids — dataset too small (n=${preparedPoints.count()}).")

    LloydContext(preparedPoints, fitDistance, initialCentroids, ownsCache)
  }

  def run(
     points: DataFrame,
     initialCentroids: Array[Vector],
     fitDistance: DistanceMetric,
     geometry: Geometry,
     maxIter: Int,
     eps: Double
  ): Array[Vector] = {
    require(initialCentroids.nonEmpty, "Lloyd.run: initial centroids must not be empty")
    val sc = points.sparkSession.sparkContext
    val kk = initialCentroids.length

    var centroids = initialCentroids
    var iteration = 0
    var hasConverged = false

    // ONE RDD fold per iteration instead of assign-UDF + groupBy/agg. This is the exception
    // rule 3(b) of the project's Spark idioms allows — "a reduction into a long array through an
    // opaque UDF" — and Lloyd is exactly that: the distance loop is opaque to Catalyst either
    // way, so the DataFrame form buys no codegen while paying a VectorUDT round-trip and a
    // vector allocation PER ROW PER ITERATION, plus a shuffle to group k keys that partial
    // aggregation has already reduced to k rows per partition.
    //
    // The accumulator is cluster-major, d sums followed by the mass: layout k*(d+1), which is
    // small (k=10, d=8 -> 90 doubles) and travels once per partition, not once per row.
    val rdd = Weights.toRdd(points)
    val dim = initialCentroids.head.size
    val accLength = kk * (dim + 1)

    while (!hasConverged && iteration < maxIter) {
      // Broadcast raw coordinates, not vectors: unpacked ONCE per round on the driver, so the
      // per-row scan runs the array kernel with its shrinking-bound early exit instead of
      // dereferencing a wrapper and re-dispatching on the vector's type at every comparison.
      val bc   = sc.broadcast(centroids.map(_.toArray))
      val dist = fitDistance

      // Plain treeAggregate: bit-for-bit reproducibility is no longer required here (reversed
      // 11.09.2026, see CLAUDE.md) — not worth the ordered variant's driver-collect cost.
      val acc = PartitionAggregator.aggregateDoubles(rdd, accLength) { (arr, row) =>
        // `toArray` is the vector's OWN array when dense — one field read per row, no copy.
        val coords = row._1.toArray
        val w      = row._2
        val c      = NearestPrototypeModel.nearestRaw(coords, bc.value, dist)
        val base   = c * (dim + 1)
        var i = 0
        while (i < dim) { arr(base + i) += w * coords(i); i += 1 }
        arr(base + dim) += w
      }

      bc.unpersist()

      // Weighted mean: a point of weight w counts as w points, so a coreset yields the centroid
      // of the data it stands for. Guarded on MASS, not on row count: a cell can hold rows whose
      // weights are all zero, and dividing by that would produce NaN centroids that then swallow
      // every point. Same guard as the Flink side's RoundStats.means.
      val newCentroids = Array.tabulate(kk) { i =>
        val base = i * (dim + 1)
        val mass = acc(base + dim)
        if (mass == 0.0) centroids(i)   // fallback for empty clusters
        else {
          val mean = new Array[Double](dim)
          var j = 0
          while (j < dim) { mean(j) = acc(base + j) / mass; j += 1 }
          geometry.project(Vectors.dense(mean))
        }
      }

      hasConverged = Convergence.hasConverged(centroids, newCentroids, eps, fitDistance)
      centroids = newCentroids
      iteration += 1
    }

    centroids
  }

  /** Seeded weighted sample of `count` distinct-ish starting centroids, taken from prepared data.
   *  Kept here so [[KMeans]] and [[BreathingKMeans]] initialise identically.
   *
   *  A-Res weighted reservoir sampling: score = `rand(seed)^(1/weight)`, take the `count`
   *  highest. With `weight = 1.0` everywhere this collapses to plain uniform sampling by
   *  `rand(seed)`, so unweighted runs are unaffected — one code path covers both, consistent
   *  with the "weighting == duplication" invariant (see [[clustering.core.Weights]]).
   */
  def sampleInitialCentroids(points: DataFrame, count: Int, seed: Long): Array[Vector] =
    points
      .withColumn("_key", pow(rand(seed), lit(1.0) / col(Columns.Weight)))
      .orderBy(desc("_key"))
      .limit(count)
      .collect()
      .map(_.getAs[Vector](Columns.Features))
}
