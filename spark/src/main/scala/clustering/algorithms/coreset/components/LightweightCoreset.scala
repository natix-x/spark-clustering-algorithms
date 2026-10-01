package clustering.algorithms.coreset.components

import clustering.core.{Columns, Weights}
import clustering.distance.DistanceMetric
import clustering.utils.PartitionAggregator
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.log4s.getLogger

/** Lightweight coreset construction (Bachem, Lucic & Krause, **KDD 2018**).
 *
 *  A coreset is a small WEIGHTED subset on which the clustering cost of any candidate solution
 *  approximates the cost on the full data. The construction is importance sampling against the
 *  distance to the mean:
 *
 *  {{{
 *  q(x) = w(x) * ( 1/(2M) + d(x, mu)^2 / (2S) )      M = total mass, S = sum_x w(x) d(x, mu)^2
 *  }}}
 *
 *  with `sum_x q(x) = 1` by construction, so `q` is a distribution over the data. Unweighted input
 *  (`w = 1`, `M = n`) gives back the paper's `1/(2n) + d^2/(2 sum d^2)` exactly.
 *
 *  ==Three passes, not two==
 *  The paper counts two, because it treats `q` as computable once `mu` is known. It is not: `q`
 *  needs the NORMALISER `S`, which is itself a full-data sum over the same distances. So the
 *  construction here is `mu` (pass 1), `S` (pass 2), draw (pass 3). Keeping `d(x, mu)` from pass 2
 *  would remove pass 3 and cost one double per row of the entire dataset — 3.5 GB on Gaia — so it
 *  is recomputed instead. The alternative that really does fit in two passes is weighted reservoir
 *  sampling on the UNNORMALISED `q`, which samples WITHOUT replacement and therefore changes the
 *  estimator; not taken, so that the weights stay the paper's.
 *
 *  ==The draw is Poisson (Bernoulli per row), not i.i.d. with replacement==
 *  The paper draws `m` points i.i.d. from `q`. Distributed, that needs a multinomial over n rows,
 *  i.e. coordination between partitions. Instead every row is drawn independently with
 *  `pi(x) = min(1, m * q(x))` and rescaled to `w_new(x) = w(x) / pi(x)`, which is the
 *  Horvitz-Thompson estimator: for any function f,
 *  `E[ sum_selected w_new f ] = sum_x pi(x) * (w(x)/pi(x)) * f(x) = sum_x w(x) f(x)`, so the
 *  coreset is an unbiased estimator of every cluster cost, exactly as the i.i.d. version is, and
 *  for unweighted data with `pi < 1` the weight is the paper's `1 / (m q(x))` unchanged.
 *
 *  The price is that the SIZE is `Binomial`-ish around `m` rather than equal to it — the same
 *  deviation the silhouette's draw has, and reported the same way ([[Result.size]] travels into the
 *  fit log and the model). A row with `m * q(x) >= 1` is taken with certainty and keeps its own
 *  weight, which is the correct treatment of a point too heavy to sample: it represents itself.
 */
private[coreset] object LightweightCoreset {

  private val logger = getLogger

  /** The drawn coreset plus the two full-data statistics it was drawn against.
   *
   *  @param points     coordinates, driver-local (`size` x d)
   *  @param weights    Horvitz-Thompson weights, index-aligned with `points`
   *  @param mass       total mass of the FULL data (`sum w`), which the weights reproduce in
   *                    expectation — the cheapest available check that the draw is sane
   *  @param dispersion `S = sum_x w(x) d(x, mu)^2` on the full data
   */
  final case class Result(
    points: Array[Array[Double]],
    weights: Array[Double],
    mass: Double,
    dispersion: Double
  ) {
    def size: Int = points.length
    def coresetMass: Double = weights.sum
  }

  /** Builds a coreset of expected size `m` from `data`, a frame of `[features, weight]`.
   *
   *  `data` is read three times, so the CALLER must have it cached (see
   *  [[clustering.core.Clusterer.fit]]). Nothing is persisted here. */
  def build(data: DataFrame, m: Int, distance: DistanceMetric, seed: Long): Result = {
    require(m >= 1, s"coreset size m must be >= 1, got $m")
    val startNanos = System.nanoTime()

    // limit-1 job, not a scan: the fold below needs a fixed accumulator length.
    val dim = data.head().getAs[Vector](Columns.Features).size
    val rdd = Weights.toRdd(data)

    // Pass 1 — mass and the weighted coordinate sums, i.e. mu. Accumulator d+1, unordered:
    // rule 3(b)'s shape (a fixed-length array behind an opaque per-row loop), and the mean only
    // feeds a sampling distribution, so last-bit merge order is immaterial.
    val moments = PartitionAggregator.aggregateDoubles(rdd, dim + 1) { (acc, row) =>
      val coords = row._1.toArray
      val w = row._2
      var i = 0
      while (i < dim) { acc(i) += w * coords(i); i += 1 }
      acc(dim) += w
    }
    val mass = moments(dim)
    require(mass > 0.0, "coreset: total mass is 0 — every row has a zero or invalid weight")
    val mean = Array.tabulate(dim)(i => moments(i) / mass)

    // Pass 2 — the normaliser S. One double, but it cannot be folded into pass 1: it is a
    // distance to a mean pass 1 is still computing.
    val metric = distance
    val dispersion = PartitionAggregator.aggregateDoubles(rdd, 1) { (acc, row) =>
      val d = metric.compute(row._1.toArray, mean)
      acc(0) += row._2 * d * d
    }.head

    // Pass 3 — the draw. `pi` is a per-row UDF (a distance loop, opaque to Catalyst either way);
    // the coin is Catalyst's own `rand`, seeded, so the draw is one expression and no row is
    // shipped to be rejected on the driver.
    val sampleSize = m
    val totalMass = mass
    val totalDispersion = dispersion
    val inclusionUdf = udf { (features: Vector, w: Double) =>
      LightweightCoreset.inclusionProbability(
        features.toArray, w, mean, totalMass, totalDispersion, sampleSize, metric)
    }

    val drawn = data
      .withColumn(InclusionColumn, inclusionUdf(col(Columns.Features), col(Columns.Weight)))
      .filter(rand(seed) < col(InclusionColumn))
      .select(col(Columns.Features), (col(Columns.Weight) / col(InclusionColumn)).as(Columns.Weight))
      .collect()

    val result = Result(
      points = drawn.map(_.getAs[Vector](Columns.Features).toArray),
      weights = drawn.map(_.getDouble(1)),
      mass = mass,
      dispersion = dispersion
    )

    val durationMs = (System.nanoTime() - startNanos) / 1000000L
    logger.info(f"coreset: built size=${result.size} requested=$m dim=$dim mass=$mass%.4f " +
      f"coresetMass=${result.coresetMass}%.4f dispersion=$dispersion%.4f seed=$seed in ${durationMs}ms")
    result
  }

  private val InclusionColumn = "_coresetInclusion"

  /** `pi(x) = min(1, m * q(x))`, the probability row `x` enters the coreset.
   *
   *  Degenerate data (every point on the mean, so `S = 0`) falls back to sampling proportional to
   *  MASS alone, which is what `q` becomes in the limit — the uniform half of the mixture. */
  private[coreset] def inclusionProbability(
    coordinates: Array[Double],
    weight: Double,
    mean: Array[Double],
    mass: Double,
    dispersion: Double,
    m: Int,
    distance: DistanceMetric
  ): Double = {
    val w = Weights.sanitize(weight)
    if (w <= 0.0) return 0.0
    val q =
      if (dispersion > 0.0) {
        val d = distance.compute(coordinates, mean)
        w * (0.5 / mass + 0.5 * d * d / dispersion)
      } else {
        w / mass
      }
    math.min(1.0, m * q)
  }
}
