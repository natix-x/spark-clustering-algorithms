package clustering.algorithms.dbscan.components

import clustering.core.Columns
import clustering.distance.DistanceMetric
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col

import scala.util.Random

/** How the m candidate core points are chosen — the knob that decides how much accuracy
 *  DBSCAN++ keeps for a given cost. Rationale: `docs/dbscanpp_docs.md` §2. */
sealed trait CandidateSelectionStrategy extends Serializable {
  def strategyName: String

  /** Selects at most `m` candidates from `data` (`n` rows, `features` column). Selection is over
   *  ROWS, not weight-proportional — densities are counted with weights afterwards. */
  def selectCandidates(
    datasetPoints: DataFrame,
    totalRowCount: Long,
    targetCandidateCount: Int,
    seed: Long,
    distanceMetric: DistanceMetric): Array[Vector]
}

/** Uniform random subsample — the paper's O(n) strategy. Fraction is oversampled and trimmed to
 *  exactly m; see `docs/dbscanpp_docs.md` §2 for why. */
object UniformSelection extends CandidateSelectionStrategy {
  val strategyName = "uniform"

  override def selectCandidates(
    datasetPoints: DataFrame,
    totalRowCount: Long,
    targetCandidateCount: Int,
    seed: Long,
    distanceMetric: DistanceMetric): Array[Vector] = {
    val featureVectors = datasetPoints.select(col(Columns.Features))
    if (targetCandidateCount >= totalRowCount) featureVectors.collect().map(_.getAs[Vector](Columns.Features))  // s = 1.0: the oracle
    else {
      val oversampledTarget = targetCandidateCount + 3.0 * math.sqrt(targetCandidateCount.toDouble)
      val oversampledCandidates = featureVectors
        .sample(withReplacement = false, fraction = math.min(1.0, oversampledTarget / totalRowCount), seed = seed)
        .collect()
        .map(_.getAs[Vector](Columns.Features))
      if (oversampledCandidates.length <= targetCandidateCount) {
        oversampledCandidates
      } else {
        val random = new Random(seed)
        val exactSubsetIndices = random.shuffle(oversampledCandidates.indices.toIndexedSeq).take(targetCandidateCount).sorted
        exactSubsetIndices.map(oversampledCandidates).toArray
      }
    }
  }
}

object CandidateSelectionStrategy {

  /** Resolves a `sampling` param value. */
  def fromName(name: String): CandidateSelectionStrategy = name.toLowerCase match {
    case UniformSelection.`strategyName` => UniformSelection
    case other => throw new IllegalArgumentException(
      s"Unknown candidate sampling: '$other'. Known: uniform")
  }
}
