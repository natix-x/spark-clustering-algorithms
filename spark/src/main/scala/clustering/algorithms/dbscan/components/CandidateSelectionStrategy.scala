package clustering.algorithms.dbscan

import clustering.core.Columns
import clustering.distance.DistanceMetric
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, monotonically_increasing_id}

import scala.util.Random

/** How the m candidate core points are chosen — the knob that decides how much accuracy
 *  DBSCAN++ keeps for a given cost. Rationale for each strategy: `docs/dbscanpp_docs.md` §2. */
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

/** Strided selection over the dataset's own order — no RNG, one pass, the cheapest of the three,
 *  but per-partition rather than global. See `docs/dbscanpp_docs.md` §2. */
object LinspaceSelection extends CandidateSelectionStrategy {
  val strategyName = "linspace"

  override def selectCandidates(
    datasetPoints: DataFrame,
    totalRowCount: Long,
    targetCandidateCount: Int,
    seed: Long,
    distanceMetric: DistanceMetric): Array[Vector] = {
    val featureVectors = datasetPoints.select(col(Columns.Features))
    if (targetCandidateCount >= totalRowCount) {
      featureVectors.collect().map(_.getAs[Vector](Columns.Features))
    } else {
      val withId = featureVectors.withColumn("rowId", monotonically_increasing_id())
      val kept =
        if (2L * targetCandidateCount <= totalRowCount) {
          withId.filter(col("rowId") % (totalRowCount / targetCandidateCount) === 0L)                     // ≥ m, < 2m
        } else {
          withId.filter(col("rowId") % math.ceil(totalRowCount.toDouble / (totalRowCount - targetCandidateCount)).toLong =!= 0L)
        }  // ≥ m, ≤ n < 2m
      val rows = kept.select(col(Columns.Features)).collect().map(_.getAs[Vector](Columns.Features))
      if (rows.length <= targetCandidateCount) {
        rows
      } else {
        Array.tabulate(targetCandidateCount)(i => rows((i.toLong * rows.length / targetCandidateCount).toInt))
      }
    }
  }
}

/** Greedy K-center (farthest-first traversal): repeatedly add the point farthest from those
 *  chosen so far — a 2-approximation of the minimax radius, run over a driver-local pool rather
 *  than the full dataset. Deviation from the paper and the `poolFactor` knob: `docs/dbscanpp_docs.md` §2. */
final class KCenterSelectionStrategy(val poolFactor: Int = 4) extends CandidateSelectionStrategy {
  val strategyName = "kcenter"

  require(poolFactor >= 1, s"poolFactor must be >= 1, got $poolFactor")

  override def selectCandidates(
    datasetPoints: DataFrame,
    totalRowCount: Long,
    targetCandidateCount: Int,
    seed: Long,
    distanceMetric: DistanceMetric): Array[Vector] = {
    val oversampledPoolSize = math.min(totalRowCount, targetCandidateCount.toLong * poolFactor).toInt
    val candidatePoolVectors = UniformSelection.selectCandidates(datasetPoints, totalRowCount, oversampledPoolSize, seed, distanceMetric)
    if (targetCandidateCount >= candidatePoolVectors.length) {
      return candidatePoolVectors
    }

    val selectedIndices = new Array[Int](targetCandidateCount)
    val isPointAlreadySelected = Array.fill(candidatePoolVectors.length)(false)

    // Paper starts arbitrarily; fixing the start at index 0 keeps it deterministic per seed.
    selectedIndices(0) = 0
    isPointAlreadySelected(0) = true
    val minDistancesToSelectedSet = Array.tabulate(candidatePoolVectors.length)(i => distanceMetric.compute(candidatePoolVectors(0), candidatePoolVectors(i)))

    // m rounds of two O(|pool|·d) scans, all on the driver — split into disjoint, ordered
    // slices across its cores, so the traversal stays bit-identical to the serial version.
    val poolSlices = KCenterSelectionStrategy.sliceRanges(candidatePoolVectors.length)

    var chosenCount = 1
    while (chosenCount < targetCandidateCount) {
      val (_, farthestCandidateIndex) = poolSlices.par.map { case (from, until) =>
        // Farthest remaining point in this slice; ties resolve to the lowest index.
        var sliceBestIndex    = -1
        var sliceBestDistance = Double.NegativeInfinity
        var candidateIndex    = from
        while (candidateIndex < until) {
          if (!isPointAlreadySelected(candidateIndex) && minDistancesToSelectedSet(candidateIndex) > sliceBestDistance) {
            sliceBestDistance = minDistancesToSelectedSet(candidateIndex)
            sliceBestIndex    = candidateIndex
          }
          candidateIndex += 1
        }
        (sliceBestDistance, sliceBestIndex)
        // `.seq` on purpose: ParSeq.reduce only contracts associativity, and the tie rule
        // needs left-to-right order. The reduce is over ≤ #cores elements, so it is free.
      }.seq.reduce(KCenterSelectionStrategy.pickFarther)

      // Only reachable when every remaining distance is NaN (NaN fails every `>`), so report
      // the cause instead of an index-out-of-bounds.
      require(farthestCandidateIndex >= 0,
        s"kcenter: all distances from the selected set are NaN after $chosenCount centres — " +
          s"the ${distanceMetric.getClass.getSimpleName} data contains NaN/Inf coordinates " +
          "(or a zero vector under cosine distance)")
      selectedIndices(chosenCount) = farthestCandidateIndex
      isPointAlreadySelected(farthestCandidateIndex) = true

      poolSlices.par.foreach { case (from, until) =>
        var candidateIndex = from
        while (candidateIndex < until) {
          val distanceToNewCenter = distanceMetric.compute(candidatePoolVectors(farthestCandidateIndex), candidatePoolVectors(candidateIndex))
          if (distanceToNewCenter < minDistancesToSelectedSet(candidateIndex)) {
            minDistancesToSelectedSet(candidateIndex) = distanceToNewCenter
          }
          candidateIndex += 1
        }
      }
      chosenCount += 1
    }

    selectedIndices.map(candidatePoolVectors)
  }
}

object KCenterSelectionStrategy {

  /** Below this the fork-join split costs more than the scan it saves. */
  private val MinParallelPoolSize = 4096

  /** Contiguous, disjoint, ascending slices of `[0, length)`; one slice below the threshold. */
  private[dbscan] def sliceRanges(length: Int): IndexedSeq[(Int, Int)] = {
    val sliceCount = if (length < MinParallelPoolSize) 1 else math.max(1, Runtime.getRuntime.availableProcessors)
    val sliceSize  = (length + sliceCount - 1) / sliceCount
    (0 until sliceCount)
      .map(slice => (slice * sliceSize, math.min(length, (slice + 1) * sliceSize)))
      .filter { case (from, until) => from < until }
  }

  /** Combines two slice argmaxes, keeping the LEFT on a tie. Slices reduce in ascending
   *  order, so the winner is the lowest index — the serial tie rule. */
  private[dbscan] def pickFarther(left: (Double, Int), right: (Double, Int)): (Double, Int) =
    if (left._2 < 0) right
    else if (right._2 < 0) left
    else if (right._1 > left._1) right
    else left
}

object CandidateSelectionStrategy {

  /** Resolves a `sampling` param value. `kcenter` also reads `poolFactor`. */
  def fromName(name: String, poolFactor: Int): CandidateSelectionStrategy = name.toLowerCase match {
    case UniformSelection.`strategyName`  => UniformSelection
    case LinspaceSelection.`strategyName` => LinspaceSelection
    case "kcenter"  => new KCenterSelectionStrategy(poolFactor)
    case other  => throw new IllegalArgumentException(
      s"Unknown candidate sampling: '$other'. Known: kcenter, linspace, uniform")
  }
}
