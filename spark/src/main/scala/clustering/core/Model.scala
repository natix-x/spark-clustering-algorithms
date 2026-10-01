package clustering.core

import org.apache.spark.sql.DataFrame


/** A fitted clustering model.
 *
 *  The sole contract is [[assignClusters]]: given a DataFrame with a
 *  [[Columns.Features]] column, return it with a [[Columns.Prediction]] column
 *  holding each row's cluster id.
 *
 *  Single-point prediction is intentionally NOT part of the contract — some
 *  models (e.g. DBSCAN) only support batch labelling via a distributed join and
 *  have no meaningful per-point predict.
 */
trait Model extends Serializable {
  def assignClusters(data: DataFrame): DataFrame

  /** How many iterations the fit's convergence loop actually ran.
   *
   *  A `maxIter` knob is a CAP, not a count: Lloyd stops as soon as the centroids move less than
   *  `tolerance`, and the medoid ladder stops when no swap improves the cost. Two runs of one
   *  configuration can therefore do very different amounts of work — measured on Ares 12.09.2026,
   *  the same k-means config at `maxIter: 100` took 68 s in one run and 160 s in another, with CPU
   *  rising in step, because the seeding differs per run and one trajectory converged early.
   *
   *  Without this number a fit's duration is uninterpretable: it is impossible to tell whether two
   *  timings differ in SPEED or in the amount of work done, and any per-iteration cost derived by
   *  dividing by `maxIter` is dividing by a number that may never have been reached.
   *
   *  Empty for an algorithm with no convergence loop — the density entries do a fixed number of
   *  passes decided by the data size, not by a stopping rule.
   */
  def iterationsRun: Option[Int] = None

  /** Realised size of the data reduction the fit optimised against — the coreset actually drawn,
   *  the candidate core points actually selected — as opposed to the `m` the config asked for.
   *
   *  The two differ whenever the reduction is a DRAW rather than a truncation, and they differ by
   *  a random amount: `coreset`'s Poisson draw and `dbscanpp`'s Bernoulli candidate selection both
   *  produce a Binomial-ish count around the requested value (measured: 1952 rows for `m: 2000`).
   *  A quality-vs-m curve plotted against the requested value is therefore plotting against a
   *  number that was never realised — the same reason the silhouette ships
   *  `silhouetteScoredPoints` beside its `sampleSize`.
   *
   *  `m` is the benchmark's universal accuracy-vs-cost knob across slots 3, 6, 8 and 9, so this is
   *  deliberately ONE generic column rather than a per-algorithm field: the landmark set and the
   *  one-batch reference set fill the same column when those slots land.
   *
   *  None for an algorithm that optimises against the full data (every k-means and medoid entry). */
  def reductionSize: Option[Int] = None
}
