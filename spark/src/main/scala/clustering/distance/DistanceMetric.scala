package clustering.distance

import org.apache.spark.ml.linalg.{Vector, Vectors}


/** A distance between two feature vectors, pluggable per run via `DistanceRegistry`.
 *
 *  ==Implementations must be pure and stateless==
 *  `compute` is called concurrently: from many Spark tasks, and — in the driver-local
 *  phases of the density and medoid algorithms — from several threads of one JVM at once
 *  (the ε-graph in `DBSCANpp` and the k-center traversal both scan in parallel). It must
 *  therefore hold no state between calls: no `var` fields, and in particular **no reusable
 *  scratch buffer**, which is the usual way an optimised metric acquires state. A buffer
 *  shared between threads would not throw — it would silently return wrong distances, and
 *  a wrong distance here is a wrong core point, i.e. a wrong cluster count.
 *
 *  Allocate locals inside `compute` instead. All current implementations do, and none
 *  allocates at all except `ManhattanDistance` (`toArray`, which on a `DenseVector` returns
 *  the vector's own array — read it, never write it).
 */
trait DistanceMetric extends Serializable {
  def compute(a: Vector, b: Vector): Double

  /** `d(a, b)` on raw coordinate arrays — the FULL distance, for callers that need the value and
   *  have no bound to exploit (pairwise matrices, medoid cost folds, BUILD/SWAP accumulators).
   *
   *  Same motivation as [[withinRadius]] minus the early exit: the `Vector` form pays a
   *  dense/sparse type match inside the library call plus a wrapper dereference per comparison,
   *  and for the Euclidean metric it also pays a `sqrt` that the callers above almost never need
   *  in the loop itself. Coordinates are unpacked ONCE per row or per collected point by the
   *  caller, never per comparison.
   *
   *  This is the signature the Flink engine's metric seam has natively, so the two engines run the
   *  same inner loop and the cross-engine timings lose a confounder. Must agree with
   *  [[compute(Vector,Vector)]] exactly.
   */
  def compute(a: Array[Double], b: Array[Double]): Double =
    compute(Vectors.dense(a), Vectors.dense(b))

  /** `d(a, b) <= radius` on raw coordinate arrays, for callers that only need the PREDICATE and
   *  never the distance — the ε-scans of `dbscanpp` (n·m in step 2, m²/2 in step 3) do nothing else.
   *
   *  Two reasons it is not `compute(...) <= radius`. A metric can skip the `sqrt`, and — the part
   *  that actually matters — it can exit the coordinate loop the moment the partial sum passes the
   *  radius. In an ε-scan almost no pair is within ε, so most comparisons end after a coordinate or
   *  two whatever the dimensionality, which is what makes these scans affordable at 256–1024 dims.
   *  Measured (11.08.2026), 2·10⁶ points × 6 000 candidates, 3-D, 8 local cores: 13.9 s → 2.7 s
   *  (5.0×), identical counts. Dropping only the `sqrt` was worth 2–10% of that, so the win is the
   *  raw-array loop plus the early exit.
   *
   *  Raw arrays rather than `Vector` on purpose: the `Vector` form pays a type match inside
   *  `Vectors.sqdist` plus a wrapper dereference per comparison, neither of which the JIT hoists out
   *  of a megamorphic call site. It is also the shape the Flink engine's metric seam has natively,
   *  so the two engines run the same inner loop and the cross-engine timings lose a confounder.
   *
   *  Callers hand over `Vector.toArray`, which is the vector's OWN array for a `DenseVector` (no
   *  copy) and one densified copy for a `SparseVector` — so this is correct for both, and paid once
   *  per row rather than per comparison. Every dataset in this thesis is dense.
   *
   *  Must agree with [[compute]] exactly, including at the boundary: `<=` is what makes a point ON
   *  the ε-sphere a neighbour. Squaring is exact for the comparison (for non-negative x and r,
   *  `x <= r  <=>  x² <= r²`). NaN coordinates make both forms false, as they do in `compute`.
   */
  def withinRadius(a: Array[Double], b: Array[Double], radius: Double): Boolean =
    compute(Vectors.dense(a), Vectors.dense(b)) <= radius

  /** `d(a, b)` when it is `<= bound`, else `Double.PositiveInfinity` — the NEAREST-NEIGHBOUR
   *  counterpart of [[withinRadius]], for a scan that keeps a running minimum.
   *
   *  Same early exit, with a bound that SHRINKS: a caller passes `min(eps, bestSoFar)`, so every
   *  hit tightens the test for the rest of the scan. Labelling in `CoreLabelModel` is the caller
   *  that matters — one pass of n·m distances over the full data, the cost that made evaluation
   *  comparable to the fit on a large core set.
   *
   *  Why returning infinity rather than the true distance is not a loss of information: the caller
   *  is looking for the minimum, and anything above the bound cannot be it. In particular, when
   *  `requireWithinEps` holds, a point whose nearest core lies beyond ε is noise whichever core
   *  that is — so restricting the scan to ε gives exactly the same labels, not an approximation.
   *
   *  Must agree with [[compute]] whenever it returns a finite value, boundary included
   *  (`d == bound` is a hit). NaN coordinates yield infinity, matching `withinRadius`'s `false`.
   */
  def distanceUpTo(a: Array[Double], b: Array[Double], bound: Double): Double = {
    val d = compute(Vectors.dense(a), Vectors.dense(b))
    if (d <= bound) d else Double.PositiveInfinity
  }

  /** Same early-exit scan as [[distanceUpTo]], for callers that only need ORDER, never the value
   *  (`NearestPrototypeModel.nearestRaw`). `bound`/return are in whatever units this metric's
   *  ordering is monotonic under — only meaningful across calls on the SAME metric instance.
   *
   *  Default: delegates to [[distanceUpTo]] — correct everywhere, but leaves `sqrt` unpruned.
   *  [[EuclideanDistance]] overrides to stay in squared space, since callers here only compare. */
  def distanceUpToOrdinal(a: Array[Double], b: Array[Double], bound: Double): Double =
    distanceUpTo(a, b, bound)
}
