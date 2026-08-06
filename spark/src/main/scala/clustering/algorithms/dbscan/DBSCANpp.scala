package clustering.algorithms.dbscan

import clustering.core.{Clusterer, Weights}
import clustering.distance.{DistanceMetric, EuclideanDistance}
import clustering.utils.UnionFind
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.DataFrame
import org.apache.spark.storage.StorageLevel
import org.log4s.getLogger

import scala.collection.mutable

/** DBSCAN++ (Jang & Jiang, *Sub-sampled DBSCAN*, ICML 2019).
 *
 *  Shrinks the GRAPH, not the data: only m = ⌈s·n⌉ sampled candidates can become core
 *  points, so components fit on the driver. Cost O(n·m) instead of O(n²).
 *
 *  Four steps:
 *   1. sample m candidates ([[CandidateSelectionStrategy]]);
 *   2. count each candidate's ε-neighbours against the FULL dataset
 *      ([[EpsilonNeighbourCounter]]) — exact densities; core point iff weighted count ≥
 *      `minPts`;
 *   3. connected components of the ε-graph over core points, driver-local ([[UnionFind]]),
 *      O(m²·d);
 *   4. label remaining points by nearest core point ([[CoreLabelModel]]).
 *
 *  With `coreSampleFraction = 1.0` and `requireWithinEps = true` the core set, cluster count
 *  and noise set are exactly classic DBSCAN. Border points within ε of two components go to
 *  the NEAREST core (DBSCAN itself picks by scan order) — deterministic by choice.
 *
 *  Cluster ids come from the smallest candidate index in each component, so reproducible.
 *
 *  @param coreSampleFraction s ∈ (0, 1]; the accuracy-vs-cost knob (m = ⌈s·n⌉)
 *  @param requireWithinEps   `true` = classic DBSCAN noise semantics, `false` = the paper's
 *                            no-noise assignment (see [[CoreLabelModel]])
 *  @param chunkSize          candidates broadcast per counting job
 */
class DBSCANpp(
  val eps: Double,
  val minPts: Int,
  val coreSampleFraction: Double,
  val samplingStrategy: CandidateSelectionStrategy = UniformSelection,
  val requireWithinEps: Boolean = true,
  val chunkSize: Int = 2000,
  val distanceMetric: DistanceMetric = EuclideanDistance,
  val seed: Long = 42L
) extends Clusterer {

  private val logger = getLogger

  require(eps > 0.0 && !eps.isNaN, s"eps must be > 0, got $eps")
  require(minPts >= 1, s"minPts must be >= 1, got $minPts")
  require(coreSampleFraction > 0.0 && coreSampleFraction <= 1.0,
    s"coreSampleFraction must be in (0, 1], got $coreSampleFraction")
  require(chunkSize >= 1, s"chunkSize must be >= 1, got $chunkSize")

  override def fit(data: DataFrame): CoreLabelModel = {
    // Cached because every counting chunk is another full pass over these rows.
    val datasetPoints = Weights.withWeights(data).persist(StorageLevel.MEMORY_AND_DISK)
    val datasetSize = datasetPoints.count()

    // m is an Int — every downstream structure (candidates, ε-graph, union-find) is driver-local.
    val candidateCount =
      math.max(1L, math.min(datasetSize, math.ceil(coreSampleFraction * datasetSize).toLong)).toInt

    // Step 1: candidate core points (P2).
    val candidatePoints =
      samplingStrategy.selectCandidates(datasetPoints, datasetSize, candidateCount, seed, distanceMetric)
    logger.info(s"dbscanpp: datasetSize=$datasetSize m=${candidatePoints.length} sampling=${samplingStrategy.strategyName} " +
      s"eps=$eps minPts=$minPts s=$coreSampleFraction")

    // Step 2: exact ε-neighbour counts against the full dataset (P1).
    val neighbourhoodCounts = EpsilonNeighbourCounter.computeNeighbourhoodDensities(
      Weights.toRdd(datasetPoints), candidatePoints, eps, distanceMetric, chunkSize)
    datasetPoints.unpersist(blocking = false)

    // Step 3: core points, in ascending candidate order — the order that fixes cluster ids.
    val coreIndices = candidatePoints.indices.filter(i => neighbourhoodCounts(i) >= minPts.toDouble).toArray
    val corePoints = coreIndices.map(candidatePoints)
    logger.info(s"dbscanpp: ${corePoints.length} of ${candidatePoints.length} candidates are core points")

    if (corePoints.isEmpty) {
      logger.warn(s"dbscanpp: no core point found (eps=$eps, minPts=$minPts) — every point is noise")
      return new CoreLabelModel(Array.empty, Array.empty, eps, distanceMetric, requireWithinEps)
    }

    // Step 4: ε-graph among the corePoints + connected components, on the driver.
    val clusterLabels = computeComponentLabels(corePoints)
    logger.info(s"dbscanpp: ${clusterLabels.distinct.length} clusters")

    new CoreLabelModel(corePoints, clusterLabels, eps, distanceMetric, requireWithinEps)
  }

  /** Connected components of the ε-graph over `cores`, as contiguous cluster ids numbered by
   *  ascending smallest member index. O(m²·d) driver work — the reason m is capped.
   *
   *  Distances run in parallel on the driver's cores; unions stay sequential in ascending
   *  `(i, j)` order ([[UnionFind]] is not thread-safe and ids depend on order). Edges are
   *  collected per BLOCK of rows to bound the edge list at `BlockSize · m` instead of O(m²).
   */
  private def computeComponentLabels(cores: Array[Vector]): Array[Int] = {
    val unionFind = new UnionFind(cores.length)
    var blockStart = 0

    // This loop produces no Spark jobs, so an empty UI and a silent log look exactly like a
    // hang — at m = 2·10⁶ it runs for hours. Throttled by time rather than by block count:
    // a block is milliseconds at m = 1 000 and minutes at m = 2·10⁶, so any fixed stride is
    // either silent or spam. Progress is measured in PAIRS, since row i scans m − i others
    // and a row percentage would run far ahead of the truth.
    val totalPairs = cores.length.toDouble * cores.length / 2
    val startedAt  = System.nanoTime()
    var lastLogAt  = startedAt

    while (blockStart < cores.length) {
      val blockEnd = math.min(blockStart + DBSCANpp.BlockSize, cores.length)
      val neighboursPerRow = (blockStart until blockEnd).par.map { i =>
        val neighbours = mutable.ArrayBuilder.make[Int]
        var j = i + 1
        while (j < cores.length) {
          if (distanceMetric.compute(cores(i), cores(j)) <= eps) neighbours += j
          j += 1
        }
        neighbours.result()
      }.toArray

      var row = blockStart
      while (row < blockEnd) {
        val neighbours = neighboursPerRow(row - blockStart)
        var t = 0
        while (t < neighbours.length) { unionFind.union(row, neighbours(t)); t += 1 }
        row += 1
      }
      blockStart = blockEnd

      val now = System.nanoTime()
      if (now - lastLogAt >= DBSCANpp.LogIntervalNanos) {
        lastLogAt = now
        val elapsed  = (now - startedAt) / 1e9
        val done     = (blockStart.toDouble * cores.length - blockStart.toDouble * blockStart / 2) / totalPairs
        logger.info(f"dbscanpp: step 3 ε-graph ${done * 100}%.1f%% (${blockStart}%d/${cores.length}%d rows), " +
          f"$elapsed%.0f s elapsed, ${elapsed * (1 - done) / done}%.0f s left")
      }
    }

    logger.info(f"dbscanpp: step 3 ε-graph done in ${(System.nanoTime() - startedAt) / 1e9}%.0f s")
    unionFind.componentIds()
  }
}

object DBSCANpp {

  /** Rows of the ε-graph scanned per parallel block. Caps the collected edge list at
   *  `BlockSize · m` ints while staying above the driver core count. */
  private val BlockSize = 128

  /** Minimum gap between two step-3 progress lines. */
  private val LogIntervalNanos = 30L * 1000000000L
}
