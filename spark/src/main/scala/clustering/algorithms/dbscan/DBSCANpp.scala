package clustering.algorithms.dbscan

import clustering.algorithms.dbscan.components.{CandidateSelectionStrategy, EpsilonGraphComponents, EpsilonNeighbourCounter, UniformSelection}
import clustering.core.{Clusterer, Weights}
import clustering.distance.{DistanceMetric, EuclideanDistance}
import org.apache.spark.sql.DataFrame
import org.log4s.getLogger


/** DBSCAN++ (Jang & Jiang, *Sub-sampled DBSCAN*, ICML 2019). Design rationale, the four steps in
 *  full, and the `dbscanexact` alias: `docs/dbscanpp_docs.md`.
 *
 *  @param coreSampleFraction s ∈ (0, 1]; the accuracy-vs-cost knob (m = ⌈s·n⌉)
 *  @param requireWithinEps   `true` = classic DBSCAN noise semantics, `false` = the paper's
 *                            no-noise assignment (see [[CoreLabelModel]])
 */
class DBSCANpp(
  val eps: Double,
  val minPts: Int,
  val coreSampleFraction: Double,
  val samplingStrategy: CandidateSelectionStrategy = UniformSelection,
  val requireWithinEps: Boolean = true,
  val distanceMetric: DistanceMetric = EuclideanDistance,
  val seed: Long = 42L
) extends Clusterer {

  private val logger = getLogger

  override def fit(data: DataFrame): CoreLabelModel = {
    val datasetPoints = Weights.withWeights(data)
    val datasetSize   = datasetPoints.count()

    // m is an Int — every downstream structure (candidates, ε-graph, union-find) is driver-local.
    val candidateCount =
      math.max(1L, math.min(datasetSize, math.ceil(coreSampleFraction * datasetSize).toLong)).toInt

    // Step 1: candidate core points
    val candidatePoints =
      samplingStrategy.selectCandidates(datasetPoints, datasetSize, candidateCount, seed, distanceMetric)
    logger.info(s"dbscanpp: datasetSize=$datasetSize m=${candidatePoints.length} sampling=${samplingStrategy.strategyName} " +
      s"eps=$eps minPts=$minPts s=$coreSampleFraction")

    // Step 2: exact ε-neighbour counts against the full dataset
    val neighbourhoodCounts = EpsilonNeighbourCounter.computeNeighbourhoodDensities(
      Weights.toRdd(datasetPoints), candidatePoints, eps, distanceMetric)

    // Core-point test (part of step 2), kept in ascending candidate order — fixes cluster ids.
    val coreIndices = candidatePoints.indices.filter(i => neighbourhoodCounts(i) >= minPts.toDouble).toArray
    val corePoints = coreIndices.map(candidatePoints)
    logger.info(s"dbscanpp: ${corePoints.length} of ${candidatePoints.length} candidates are core points")

    if (corePoints.isEmpty) {
      logger.warn(s"dbscanpp: no core point found (eps=$eps, minPts=$minPts) — every point is noise")
      return new CoreLabelModel(Array.empty, Array.empty, eps, distanceMetric, requireWithinEps)
    }

    // Step 3: ε-graph among the core points + connected components
    val clusterLabels = EpsilonGraphComponents.compute(
      corePoints, eps, distanceMetric, data.sparkSession.sparkContext)
    logger.info(s"dbscanpp: ${clusterLabels.distinct.length} clusters")

    new CoreLabelModel(corePoints, clusterLabels, eps, distanceMetric, requireWithinEps)
  }
}
