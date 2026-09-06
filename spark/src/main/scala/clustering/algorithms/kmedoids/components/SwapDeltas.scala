package clustering.algorithms.kmedoids.components


/** The FastPAM1 Δ formula (Schubert & Rousseeuw 2019): ONE pass over the points yields the cost
 *  change for swapping a candidate into EVERY medoid slot.
 *
 *    Δ(h, i)          = shared(h) + removeLoss(h)(i)
 *    shared(h)        = Σ_j w_j·[ d(h,j) < d1(j) ? d(h,j) − d1(j) : 0 ]
 *    removeLoss(h)(i) = Σ_{j: n1(j)=i} [ w_j·(min(d2(j), d(h,j)) − d1(j)) − sharedContrib_j ]
 *
 *  Derivation: `docs/kmedoids_docs.md`.
 */
private[kmedoids] object SwapDeltas {

  /** Writes Δ(candidate, slot) for every slot into `deltasBySlot` (length k). O(n). */
  def forCandidate(
    distances: DistanceMatrix,
    candidate: Int,
    weights: Array[Double],
    cache: NearestMedoidCache,
    deltasBySlot: Array[Double]
  ): Unit = {
    java.util.Arrays.fill(deltasBySlot, 0.0)

    var sharedGain = 0.0
    var point = 0
    while (point < distances.pointCount) {
      val distanceToCandidate = distances(candidate, point)
      val weight = weights(point)
      val nearest  = cache.nearestDistance(point)

      // Gain the point takes from `candidate` whichever slot is removed.
      val sharedContribution =
        if (distanceToCandidate < nearest) weight * (distanceToCandidate - nearest) else 0.0
      sharedGain += sharedContribution

      // Extra cost carried only if the point's OWN slot is the one removed: it falls back to
      // the candidate or to its second-nearest medoid, whichever is closer.
      deltasBySlot(cache.nearestSlot(point)) +=
        weight * (math.min(cache.secondNearestDistance(point), distanceToCandidate) - nearest) -
          sharedContribution

      point += 1
    }

    var slot = 0
    while (slot < deltasBySlot.length) { deltasBySlot(slot) += sharedGain; slot += 1 }
  }

  def bestMoveForCandidate(
    distances: DistanceMatrix,
    candidate: Int,
    weights: Array[Double],
    cache: NearestMedoidCache,
    deltasBySlot: Array[Double]
  ): SwapMove = {
    forCandidate(distances, candidate, weights, cache, deltasBySlot)
    var best = SwapMove.None
    var slot = 0
    while (slot < deltasBySlot.length) {
      if (deltasBySlot(slot) < 0.0) {
        best = SwapMove.preferred(best, SwapMove(deltasBySlot(slot), slot, candidate))
      }
      slot += 1
    }
    best
  }
}
