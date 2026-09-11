package clustering.algorithms.dbscan.components

import clustering.core.Columns
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col

import scala.util.Random

/** How the m candidate core points are chosen for DBSCAN++ — the knob that decides how much
 *  accuracy is kept for a given cost. Rationale: `docs/dbscanpp_docs.md` §2.
 *
 *  Uniform random subsample — the paper's O(n) strategy. Fraction is oversampled and trimmed to
 *  exactly m; see `docs/dbscanpp_docs.md` §2 for why.
 *
 *  This used to sit behind a `CandidateSelectionStrategy` trait, but uniform is the only
 *  selection ever implemented — the trait was pure indirection. If a second strategy shows up,
 *  re-extract the trait then.
 */
object UniformSelection {
  val strategyName = "uniform"

  /** Resolves a `sampling` param value. */
  def fromName(name: String): UniformSelection.type = name.toLowerCase match {
    case `strategyName` => UniformSelection
    case other => throw new IllegalArgumentException(
      s"Unknown candidate sampling: '$other'. Known: uniform")
  }

  /** Every row, unmodified — the s = 1.0 oracle (exact classic DBSCAN). Also the size-discovery
   *  step for that case: the caller needs n before it can compute anything downstream, and this
   *  same collect() already knows it, so no separate `count()` job is needed just to reach it. */
  def selectAll(datasetPoints: DataFrame): Array[Vector] =
    datasetPoints.select(col(Columns.Features)).collect().map(_.getAs[Vector](Columns.Features))

  /** Selects at most `m` candidates from `data` (`n` rows, `features` column). Selection is over
   *  ROWS, not weight-proportional — densities are counted with weights afterwards. */
  def selectCandidates(
    datasetPoints: DataFrame,
    totalRowCount: Long,
    targetCandidateCount: Int,
    seed: Long): Array[Vector] = {
    val featureVectors = datasetPoints.select(col(Columns.Features))
    if (targetCandidateCount >= totalRowCount) selectAll(datasetPoints)  // s = 1.0: the oracle
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
