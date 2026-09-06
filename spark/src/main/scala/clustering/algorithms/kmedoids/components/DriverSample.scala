package clustering.algorithms.kmedoids.components

import scala.reflect.ClassTag
import scala.util.Random

/** The two things every "collect a small random subset to the driver" site in this package needs,
 *  in one place because getting either wrong is silent.
 *
 *  Both `clara`'s samples and `pamae`'s candidate pool once ended their draw with `take(m)` on a
 *  sampled DataFrame. That looks like a random subset and is not: Spark's `take` scans partitions
 *  in order until the count is met, so the result is a PREFIX of the input. On the ordered
 *  datasets in this benchmark — Gaia by sky position, taxi trips by date — a positional bias is a
 *  distributional bias, so the sample was drawn from one region of the data while reporting itself
 *  as uniform.
 */
private[kmedoids] object DriverSample {

  /** How far to over-draw when a subset of an exact size has to be cut from a Bernoulli draw,
   *  whose count is Binomial(n, p). At 1.3 the shortfall probability is negligible for any sample
   *  size worth running, and the surplus is discarded by [[takeRandom]]. */
  val OversampleFactor = 1.3

  /** `size` elements chosen uniformly from `rows` (all of them when there are fewer).
   *
   *  Shuffles and THEN truncates. The order matters: truncating first would keep a prefix of
   *  whatever order the rows arrived in, which is the scan order — exactly the bias this object
   *  exists to prevent. */
  def takeRandom[A: ClassTag](rows: Array[A], size: Int, seed: Long): Array[A] =
    if (rows.length <= size) rows
    else new Random(seed).shuffle(rows.toIndexedSeq).take(size).toArray

  /** Inclusion probability for a Bernoulli draw meant to yield about `size` rows out of
   *  `rowCount`, with the over-draw already applied. */
  def inclusionProbability(size: Int, rowCount: Long): Double =
    math.min(1.0, OversampleFactor * size.toDouble / rowCount.toDouble)
}
