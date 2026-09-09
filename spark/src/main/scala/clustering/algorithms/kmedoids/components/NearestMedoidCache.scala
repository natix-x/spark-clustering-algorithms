package clustering.algorithms.kmedoids.components

/** Per-point distances to the nearest and second-nearest medoid, plus the SLOT (0..k-1) of each —
 *  the four arrays the FastPAM1 Δ formula ([[SwapDeltas]]) and FasterPAM's incremental update both
 *  work over (`docs/kmedoids_docs.md`).
 *
 *  Allocated once per fit and updated in place, so the swap loops allocate nothing.
 */
private[kmedoids] final class NearestMedoidCache(pointCount: Int) {

  val nearestDistance: Array[Double] = new Array[Double](pointCount)
  val secondNearestDistance: Array[Double] = new Array[Double](pointCount)
  val nearestSlot: Array[Int] = new Array[Int](pointCount)
  val secondSlot: Array[Int] = new Array[Int](pointCount)

  /** Recomputes all four arrays for every point, from scratch, in O(n·k). Used for the initial
   *  fill and as the rescan fallback inside [[updateAfterSwap]]. */
  def refresh(distances: DistanceMatrix, medoids: Array[Int]): Unit = {
    var point = 0
    while (point < pointCount) {
      refreshPoint(point, distances, medoids)
      point += 1
    }
  }

  private def refreshPoint(point: Int, distances: DistanceMatrix, medoids: Array[Int]): Unit = {
    val k = medoids.length
    var nearest = Double.MaxValue
    var second = Double.MaxValue
    var nearestIdx = -1
    var secondIdx = -1
    var slot = 0
    while (slot < k) {
      val d = distances(point, medoids(slot))
      if (d < nearest) { second = nearest; secondIdx = nearestIdx; nearest = d; nearestIdx = slot }
      else if (d < second) { second = d; secondIdx = slot }
      slot += 1
    }
    nearestDistance(point)       = nearest
    secondNearestDistance(point) = second
    nearestSlot(point)           = nearestIdx
    secondSlot(point)            = secondIdx
  }

  /** Updates the cache for ONE swap at `swappedSlot` — `medoids(swappedSlot)` is already the NEW
   *  medoid — in O(n) plus a full O(k) rescan for whichever points had the old medoid in their own
   *  top 2. Counterpart of the Flink side's `NearestMedoidCache.updateAfterSwap`; see there for the
   *  case-by-case argument that this is EXACT, not approximate — a point's top 2 either provably
   *  doesn't need the swapped-out medoid's identity to update in O(1), or it does and gets a full
   *  rescan, never a guess.
   *
   *  Replaces the O(n·k) [[refresh]] FasterPAM used to call after every accepted swap. The total
   *  rescan work across a pass is bounded by (roughly) the size of the cluster the swapped-out
   *  medoid served, i.e. O(n) in total when clusters are balanced — the same amortised cost the
   *  FasterPAM paper claims, and never worse than the O(n·k) this replaces. */
  def updateAfterSwap(distances: DistanceMatrix, medoids: Array[Int], swappedSlot: Int): Unit = {
    val newMedoidIdx = medoids(swappedSlot)
    var point = 0
    while (point < pointCount) {
      val candidateDist = distances(point, newMedoidIdx)
      val nearSlotHere = nearestSlot(point)
      val secSlotHere = secondSlot(point)

      if (nearSlotHere == swappedSlot) {
        if (candidateDist <= secondNearestDistance(point)) {
          // The new medoid still owns this slot as nearest; the second is untouched because the
          // medoid it names never moved. Exact, O(1).
          nearestDistance(point) = candidateDist
        } else {
          // The old second is now the true nearest, but the new second is a medoid this cache
          // never tracked (the point's "third nearest") — only a rescan is exact.
          refreshPoint(point, distances, medoids)
        }
      } else if (secSlotHere == swappedSlot && candidateDist >= nearestDistance(point)) {
        // The old second is displaced and the candidate doesn't even beat the nearest, so whether
        // it becomes the new second depends on a third medoid this cache never tracked. Rescan.
        refreshPoint(point, distances, medoids)
      } else if (candidateDist < nearestDistance(point)) {
        // swappedSlot was not in the top 2 (or was the second and just got promoted past the
        // nearest) — demoting the old nearest to second and inserting the candidate as nearest is
        // exact either way.
        secondNearestDistance(point) = nearestDistance(point)
        secondSlot(point) = nearSlotHere
        nearestDistance(point) = candidateDist
        nearestSlot(point) = swappedSlot
      } else if (candidateDist < secondNearestDistance(point)) {
        secondNearestDistance(point) = candidateDist
        secondSlot(point) = swappedSlot
      }
      // else: swappedSlot is outside this point's top 2 both before and after — unchanged.
      point += 1
    }
  }
}
