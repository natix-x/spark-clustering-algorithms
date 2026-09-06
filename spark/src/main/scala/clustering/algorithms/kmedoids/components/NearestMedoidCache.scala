package clustering.algorithms.kmedoids.components

/** Per-point distances to the nearest and second-nearest medoid, plus the SLOT (0..k-1) of the
 *  nearest one — the three arrays the FastPAM1 Δ formula is expressed in
 *  ([[SwapDeltas]], `docs/kmedoids_docs.md`).
 *
 *  Allocated once per fit and refreshed in place, so the swap loops allocate nothing.
 */
private[kmedoids] final class NearestMedoidCache(pointCount: Int) {

  val nearestDistance: Array[Double] = new Array[Double](pointCount)
  val secondNearestDistance: Array[Double] = new Array[Double](pointCount)
  val nearestSlot: Array[Int] = new Array[Int](pointCount)

  /** Recomputes all three arrays in O(n·k). */
  def refresh(distances: DistanceMatrix, medoids: Array[Int]): Unit = {
    val k = medoids.length
    var point = 0
    while (point < pointCount) {
      var nearest = Double.MaxValue
      var secondNearest = Double.MaxValue
      var nearestIndex = -1
      var slot = 0
      while (slot < k) {
        val d = distances(point, medoids(slot))
        if (d < nearest) { secondNearest = nearest; nearest = d; nearestIndex = slot }
        else if (d < secondNearest) secondNearest = d
        slot += 1
      }
      nearestDistance(point)       = nearest
      secondNearestDistance(point) = secondNearest
      nearestSlot(point)           = nearestIndex
      point += 1
    }
  }
}
