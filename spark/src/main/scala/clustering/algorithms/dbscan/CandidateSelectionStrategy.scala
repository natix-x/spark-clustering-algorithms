package clustering.algorithms.dbscan

import clustering.core.Columns
import clustering.distance.DistanceMetric
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, monotonically_increasing_id}

import scala.util.Random

/** How the m candidate core points are chosen — the knob that decides how much of the
 *  accuracy DBSCAN++ keeps for a given cost.
 *
 *  The paper (Jang & Jiang, ICML 2019) offers uniform random and greedy K-center
 *  (a 2-approximation, generally better in their experiments); `linspace` comes from the
 *  Spark DBSCAN++ realisation (IJDSA 2026) and is the cheapest of the three. All are
 *  P2: one job in DataFrame/Catalyst idioms, result collected to the driver, then broadcast by
 *  the counting job.
 */
sealed trait CandidateSelectionStrategy extends Serializable {
  def strategyName: String

  /** Selects at most `m` candidates from `data` (`n` rows, `features` column).
   *
   *  Selection is over ROWS, not weight-proportional: candidates only have to cover the space,
   *  and the densities that decide core-point status are counted with weights afterwards. */
  def selectCandidates(
    datasetPoints: DataFrame,
    totalRowCount: Long,
    targetCandidateCount: Int,
    seed: Long,
    distanceMetric: DistanceMetric): Array[Vector]
}

/** Uniform random subsample — the paper's O(n) strategy.
 *
 *  Fraction is (m + 3√m)/n, not m/n: `sample` is a per-row Bernoulli trial, so m/n returns
 *  m ± √m rows and half the seeds undershoot. The overshoot is trimmed to exactly m with a
 *  seeded RNG on the driver — not with `take(m)`, which is `CollectLimitExec` and would take
 *  the first partitions only (prefix bias on data ordered by position: Gaia, TLC).
 */
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

/** Strided selection over the dataset's own order — no RNG, one pass, the cheapest of the
 *  three. Honest weakness: on data ordered by position (Gaia by sky region, TLC by time) a
 *  stride can systematically miss regions.
 *
 *  Not a GLOBAL linspace: `monotonically_increasing_id` is `partitionId << 33 |
 *  rowInPartition`, so `id % k === 0` strides within each partition with a phase that jumps
 *  at partition boundaries. It is also a `nondeterministic` Catalyst expression — reproducible
 *  only for a fixed partitioning. A true global stride would need `zipWithIndex` or
 *  `row_number()` over an `orderBy`.
 *
 *  Two steps, not one modulo: no `rowId % k` lands on exactly m rows, and `take(m)` would add
 *  prefix bias (and for s > 0.5 the stride collapses to 1 and spreads nothing). So the filter
 *  keeps AT LEAST m rows, and the exact count comes from a linspace over the collected array.
 */
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
      // Monotone and contiguous within a partition, which is all a per-partition stride
      // needs, and it avoids an RDD zipWithIndex pass. See the class docstring for what
      // this stride is and is not.
      val withId = featureVectors.withColumn("rowId", monotonically_increasing_id())
      val kept =
        if (2L * targetCandidateCount <= totalRowCount) {
          withId.filter(col("rowId") % (totalRowCount / targetCandidateCount) === 0L)                     // ≥ m, < 2m
        } else {
          withId.filter(col("rowId") % math.ceil(totalRowCount.toDouble / (totalRowCount - targetCandidateCount)).toLong =!= 0L)
        }  // ≥ m, ≤ n < 2m
      val rows = kept.select(col(Columns.Features)).collect().map(_.getAs[Vector](Columns.Features))
      // Phase restarts per partition, so the kept count deviates by O(#partitions), not by
      // one row: a partition narrower than the stride yields 0 or 1 rows. Plot the m that
      // DBSCANpp logs, never the nominal one.
      if (rows.length <= targetCandidateCount) {
        rows
      } else {
        Array.tabulate(targetCandidateCount)(i => rows((i.toLong * rows.length / targetCandidateCount).toInt))
      }
    }
  }
}

/** Greedy K-center (farthest-first traversal): repeatedly add the point farthest from those
 *  chosen so far — a 2-approximation of the minimax radius, so coverage beats uniform.
 *
 *  **Documented deviation.** The paper traverses the full dataset, which needs m sequential
 *  distributed rounds (5 000 Spark jobs). Here it runs on the driver over a uniform POOL of
 *  `poolFactor · m` points: one job, then O(m · |pool| · d) across the driver's cores. The
 *  guarantee then holds against the pool, not the full data — say so in the thesis.
 *  `poolFactor` is a second accuracy-vs-cost knob: at 1 the strategy IS uniform, and where
 *  its advantage saturates is a cheap empirical result.
 */
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
