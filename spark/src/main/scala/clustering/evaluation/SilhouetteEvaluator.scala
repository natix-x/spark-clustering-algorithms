package clustering.evaluation

import clustering.core.{Columns, Model, Weights}
import clustering.distance.{DistanceMetric, EuclideanDistance}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{Column, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, IntegerType}

import scala.collection.mutable.ArrayBuffer

/** Mean silhouette (Rousseeuw 1987) — the only O(n²) metric in the contract, hence the only one
 *  that is subsampled.
 *
 *  Shape (P2): label, drop noise, subsample, collect survivors to the driver once, broadcast,
 *  score index ranges of that broadcast across the cluster. One data read; m² work spread over
 *  the cluster; driver only ever holds the sample.
 *
 *  Sampling happens after the noise filter, so `sampleSize` always means the number of SCORED
 *  points regardless of a run's noise fraction. Weighted throughout — weighting == duplication
 *  ([[clustering.core.Weights]]): a neighbour of weight `w` counts `w` times in `a`/`b`, cluster
 *  size is its mass. On a reduced input (coreset) this reads higher than the true silhouette
 *  (misses collapsed-neighbourhood dispersion) — validate `coreset` via ARI/NMI instead.
 *
 *  `a`/`b` are estimated from the sample: `a` is unbiased, `b` (a min over k sample means) is
 *  biased downwards, growing as `sampleSize / k` shrinks — never compare across k without holding
 *  that ratio fixed; prefer [[DaviesBouldinIndex]]/[[CalinskiHarabaszIndex]] for that axis.
 *
 *  `silhouetteScoredPoints`/`silhouetteSampleClusters`/`silhouetteUnscoredPoints` (against the
 *  full-data `nClusters`) expose two further sample effects: a cluster drawing under two sample
 *  points is excluded (not scored 0); a cluster drawing none is invisible to `b`, biasing it up —
 *  visible as `sampleClusters < nClusters`. A genuine singleton (full mass ≤ 1) still scores 0.0;
 *  telling it apart from an under-drawn cluster needs the full-data masses ([[Population]]).
 */
class SilhouetteEvaluator(val distance: DistanceMetric = EuclideanDistance) extends ClusteringEvaluator {

  /** Full-data silhouette. Only safe on inputs small enough to collect to the driver — see
   *  [[measure]], which the benchmark calls instead. */
  override def evaluate(model: Model, data: DataFrame): Double =
    measure(model, data, sampleSize = None, seed = 0L, population = None).score

  /** @param sampleSize at most this many points are scored; `None` scores everything
   *  @param population per-cluster mass of the labelled, de-noised full data, if already known —
   *                    supplies the draw's denominator and tells a true singleton apart from an
   *                    under-drawn cluster
   */
  def measure(
    model: Model,
    data: DataFrame,
    sampleSize: Option[Int],
    seed: Long,
    population: Option[SilhouetteEvaluator.Population]
  ): SilhouetteEvaluator.Outcome = {
    val sc = data.sparkSession.sparkContext

    // Checked on the input, not the projection below — the projection always carries a `weight`
    // column and would always answer "yes".
    val weighted = Weights.isWeighted(data)

    // Corrupt weights neutralised here, on the one projection both draw branches and the collect
    // read from: a null weight used to NPE on `Row.getDouble`, a negative one dragged the score
    // toward -1.
    val labeled = model.assignClusters(data)
      .select(
        col(Columns.Prediction).cast(IntegerType).as(Columns.Prediction),
        col(Columns.Features),
        Weights.safeColumn(data).as(Columns.Weight))
      .filter(col(Columns.Prediction) =!= SilhouetteEvaluator.NoiseLabel)
      .filter(col(Columns.Weight) > 0.0)

    val drawn = sampleSize.filter(_ > 0) match {
      case Some(s) => SilhouetteEvaluator.subsample(labeled, s, seed, population.map(_.totalMass), weighted)
      case None    => labeled
    }

    val collected = SilhouetteEvaluator.Sample.collectFrom(drawn, population.map(_.clusterMass))
    val sample    = collected.sample

    // Fewer than two clusters: `b` is undefined, so the score is too. Reported as 0.0 (the
    // convention the other evaluators use), not the ~1.0 a missing `b` treated as infinity would
    // give — which would hand a degenerate one-cluster labelling the best score in the matrix.
    if (sample.clusterCount < 2) {
      return SilhouetteEvaluator.Outcome(
        score = 0.0,
        scoredPoints = 0,
        sampleClusters = sample.clusterCount,
        unscoredPoints = sample.size + collected.droppedRows)
    }

    val bcSample = sc.broadcast(sample)
    val dist = distance
    val parts = math.max(1, math.min(sample.size, sc.defaultParallelism))

    try {
      // Each task folds its slice into (Σ w·s, Σ w, #scored, #unscorable) — a weighted mean,
      // independent of task finish order.
      val partials = sc.parallelize(0 until sample.size, parts)
        .mapPartitions { indices =>
          val s = bcSample.value
          val scratch = new Array[Double](s.clusterCount)
          var scoreSum = 0.0
          var massSum = 0.0
          var scored = 0
          var unscored = 0
          while (indices.hasNext) {
            val i = indices.next()
            val v = s.score(i, scratch, dist)
            if (java.lang.Double.isNaN(v)) unscored += 1
            else {
              val w = s.weights(i)
              scoreSum += w * v
              massSum += w
              scored += 1
            }
          }
          Iterator.single((scoreSum, massSum, scored, unscored))
        }
        .collect()

      val scoreSum = partials.foldLeft(0.0)(_ + _._1)
      val massSum  = partials.foldLeft(0.0)(_ + _._2)
      val scored   = partials.foldLeft(0)(_ + _._3)
      val unscored = partials.foldLeft(0)(_ + _._4)

      SilhouetteEvaluator.Outcome(
        score = if (massSum <= 0.0) 0.0 else scoreSum / massSum,
        scoredPoints = scored,
        sampleClusters = sample.clusterCount,
        unscoredPoints = unscored + collected.droppedRows)
    } finally bcSample.destroy()
  }
}

object SilhouetteEvaluator {

  private val NoiseLabel = -1

  /** Per-cluster mass of the labelled, de-noised full data — cluster label -> Σ weight. Noise
   *  (-1) must already be filtered out — `totalMass` is the draw's denominator and must match
   *  the scored population. */
  final case class Population(clusterMass: Map[Int, Double]) {
    val totalMass: Double = clusterMass.valuesIterator.sum
  }

  /** @param score          mass-weighted mean silhouette over the scored points
   *  @param scoredPoints   drawn points that got a score
   *  @param sampleClusters distinct clusters present in the sample — a shortfall vs full-data
   *                        `nClusters` means `b` was minimised over survivors only
   *  @param unscoredPoints drawn points that did not get a score: dropped for a non-finite
   *                        coordinate/weight, or in a cluster that drew fewer than two members
   */
  final case class Outcome(score: Double, scoredPoints: Int, sampleClusters: Int, unscoredPoints: Int)

  /** At most `sampleSize` points for the silhouette. Unweighted: keeps each row with probability
   *  `sampleSize / n`. Weighted: draws rows in proportion to mass, every draw unit-weighted, so
   *  the sample represents the points the frame stands for (a uniform row sample would keep or
   *  drop a heavy coreset row whole). `knownTotalMass` lets a caller that already scanned the
   *  labels skip a second pass.
   *
   *  `weighted` describes the original input and is passed in rather than sniffed from `data`:
   *  by the time a frame reaches here it always carries a `weight` column.
   *
   *  Both branches draw via [[rowUniform]] (content hash, not RNG), so the sample is a pure
   *  function of (rows, seed), independent of partition count.
   *
   *  Draw rule per row: `copies = floor(L) + [u < frac(L)]`, `L = w · sampleSize / totalMass`,
   *  `u` uniform — `E[copies] = L`, integer part goes in deterministically so the sample size
   *  concentrates on `sampleSize`.
   */
  private[evaluation] def subsample(
    data: DataFrame,
    sampleSize: Int,
    seed: Long,
    knownTotalMass: Option[Double],
    weighted: Boolean
  ): DataFrame =
    if (!weighted) {
      val total = knownTotalMass.getOrElse(data.count().toDouble)
      if (total <= sampleSize) data
      else data.filter(rowUniform(data, seed) < lit(sampleSize / total))
    } else {
      val totalMass = knownTotalMass.getOrElse(Weights.totalMass(data))
      if (totalMass <= sampleSize) data
      else {
        val lambda = Weights.safeColumn(data) * lit(sampleSize / totalMass)
        val draw = floor(lambda) + when(rowUniform(data, seed) < (lambda - floor(lambda)), 1L).otherwise(0L)

        // `coalesce` before `least`/`greatest`: those skip nulls, so a null weight without it
        // would explode into the entire sample.
        val copies = greatest(least(coalesce(draw, lit(0L)), lit(sampleSize.toLong)), lit(0L)).cast(IntegerType)

        data
          .withColumn(CopiesCol, copies)
          .withColumn(DrawCol, explode(array_repeat(lit(1.0), col(CopiesCol))))
          .withColumn(Columns.Weight, col(DrawCol))
          .drop(CopiesCol, DrawCol)
      }
    }

  /** A uniform [0, 1) draw for a row, computed from the row's content and the seed alone.
   *
   *  Replaces `rand(seed)`/`DataFrame.sample(seed)`, which seed each partition with
   *  `seed + partitionIndex` — making the sample a function of row-to-partition layout, so the
   *  reported silhouette used to drift with worker count on an otherwise identical clustering.
   *
   *  `xxhash64` over `features` (+ `weight` when present, so same-coordinate coreset rows of
   *  different mass draw independently); `shiftRightUnsigned(_, 11)` leaves 53 bits, the widest
   *  a Double holds exactly.
   *
   *  Consequence: rows agreeing on every hashed column share a draw and move as a block —
   *  harmless on embeddings, real on tabular data (grid-snapped coordinates), inflating variance
   *  (not bias) past the Binomial(n, p) size that drawing otherwise gives.
   */
  private[evaluation] def rowUniform(data: DataFrame, seed: Long): Column = {
    val hashed =
      if (Weights.isWeighted(data)) xxhash64(col(Columns.Features), col(Columns.Weight), lit(seed))
      else xxhash64(col(Columns.Features), lit(seed))
    shiftRightUnsigned(hashed, 11).cast(DoubleType) / lit((1L << 53).toDouble)
  }

  private val CopiesCol = "__copies"
  private val DrawCol   = "__draw"

  /** The collected sample, flattened into raw arrays and broadcast once — coordinates unpacked
   *  once so the inner loop is the same raw-array kernel paid n² times, not per access.
   *
   *  @param points   one row's coordinates per entry
   *  @param weights  matching weights
   *  @param slots    matching cluster slot (dense 0..k-1 index, not the cluster label)
   *  @param slotMass total weight per slot in the sample — denominator of `a`/`b`
   *  @param trueMass total weight per slot in the full data if known, else a copy of `slotMass` —
   *                  used only to tell a genuine singleton (score 0) from an under-drawn cluster
   */
  private[evaluation] final case class Sample(
    points: Array[Array[Double]],
    weights: Array[Double],
    slots: Array[Int],
    slotMass: Array[Double],
    trueMass: Array[Double]
  ) {

    def size: Int = points.length

    def clusterCount: Int = slotMass.length

    /** The silhouette of point `i`, or `NaN` when unscorable (its cluster drew fewer than two
     *  sample members). `scratch` is caller-owned, reused across a task's rows.
     *
     *  `a` needs the mean distance to OTHER members of the point's cluster: rather than track a
     *  row id to skip itself, sums over all members and removes exactly one self-contribution,
     *  `d(p,p)` subtracted rather than assumed 0 ([[clustering.distance.CosineDistance]] reports
     *  1.0 for a zero vector against itself). Denominator is `mass − min(1, w_p)`: for
     *  `w_p ≥ 1` that's exactly the mass minus one; for a fractional `w_p` a flat 1 would give a
     *  denominator smaller than the neighbours' mass.
     */
    def score(i: Int, scratch: Array[Double], dist: DistanceMetric): Double = {
      java.util.Arrays.fill(scratch, 0.0)

      val p = points(i)
      var j = 0
      while (j < points.length) {
        scratch(slots(j)) += weights(j) * dist.compute(p, points(j))
        j += 1
      }

      val own = slots(i)

      // Full mass 1 = single point, no neighbours: `a` undefined, silhouette 0 (Rousseeuw), not
      // the 1.0 treating `a` as 0 would give.
      if (trueMass(own) <= 1.0) return 0.0

      // A real cluster the sample drew fewer than two members of is unscorable, not a 0 — that
      // would be a second, discrete downward bias on top of min-of-noisy-b.
      val self  = math.min(1.0, weights(i))
      val denom = slotMass(own) - self
      if (denom <= 0.0) return Double.NaN

      val a = (scratch(own) - self * dist.compute(p, p)) / denom

      var b = Double.PositiveInfinity
      var slot = 0
      while (slot < scratch.length) {
        if (slot != own && slotMass(slot) > 0.0) {
          val mean = scratch(slot) / slotMass(slot)
          if (java.lang.Double.isFinite(mean) && mean < b) b = mean
        }
        slot += 1
      }

      val denominator = math.max(a, b)
      if (!java.lang.Double.isFinite(denominator) || denominator <= 0.0) 0.0
      else {
        val s = (b - a) / denominator
        if (java.lang.Double.isFinite(s)) s else 0.0
      }
    }
  }

  /** A collected [[Sample]] plus the number of drawn rows it refused. */
  private[evaluation] final case class Collected(sample: Sample, droppedRows: Int)

  private[evaluation] object Sample {

    /** Collect the drawn rows and drop the ones that cannot take part in a distance: one
     *  non-finite coordinate poisons its cluster's distance sum to NaN for every other point,
     *  which never wins `min`, silently inflating neighbours' `b` from a far-away cluster
     *  instead. Filtered here (O(sample·d) on driver), and the drop count is reported.
     *
     *  Fields read by name, not positionally — safer against `subsample`'s column order.
     */
    def collectFrom(labeled: DataFrame, trueClusterMass: Option[Map[Int, Double]]): Collected = {
      val rows = labeled.collect()

      val labelBuf  = new ArrayBuffer[Int](rows.length)
      val pointBuf  = new ArrayBuffer[Array[Double]](rows.length)
      val weightBuf = new ArrayBuffer[Double](rows.length)
      var dropped = 0

      var i = 0
      while (i < rows.length) {
        val row = rows(i)
        val w = row.getAs[Double](Columns.Weight)
        val v = row.getAs[Vector](Columns.Features).toArray // the vector's own array when dense
        if (java.lang.Double.isFinite(w) && w > 0.0 && allFinite(v)) {
          labelBuf += row.getAs[Int](Columns.Prediction)
          pointBuf += v
          weightBuf += w
        } else dropped += 1
        i += 1
      }

      val labels  = labelBuf.toArray
      val points  = pointBuf.toArray
      val weights = weightBuf.toArray

      // Ascending label order: score independent of partition arrival order.
      val slotOf = labels.distinct.sorted.zipWithIndex.toMap
      val slots  = labels.map(slotOf)

      val slotMass = new Array[Double](slotOf.size)
      i = 0
      while (i < slots.length) { slotMass(slots(i)) += weights(i); i += 1 }

      // Full-data mass per slot when known; else the sample's own mass (can't then tell a true
      // singleton from an under-drawn cluster).
      val trueMass = new Array[Double](slotOf.size)
      slotOf.foreach { case (label, slot) =>
        trueMass(slot) = trueClusterMass.flatMap(_.get(label)).getOrElse(slotMass(slot))
      }

      Collected(Sample(points, weights, slots, slotMass, trueMass), dropped)
    }

    private def allFinite(xs: Array[Double]): Boolean = {
      var i = 0
      while (i < xs.length) {
        if (!java.lang.Double.isFinite(xs(i))) return false
        i += 1
      }
      true
    }
  }
}