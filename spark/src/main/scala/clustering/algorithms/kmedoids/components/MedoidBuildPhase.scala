package clustering.algorithms.kmedoids.components

import clustering.utils.DriverParallelism

/** BUILD phase (Kaufman & Rousseeuw 1990): greedy selection of k initial medoids, shared by every
 *  driver-local rung so they differ only in their SWAP strategy.
 *
 *  Both scans of every round run over disjoint ascending slices of the point range and reduce
 *  left-to-right, so the parallel result is the serial one — see `docs/kmedoids_docs.md`.
 */
private[kmedoids] object MedoidBuildPhase {

  /** @param weights `weights(j)` = how many points `j` stands for
   *  @return indices of the k initial medoids, in selection order */
  def selectInitialMedoids(distances: DistanceMatrix, k: Int, weights: Array[Double]): Array[Int] = {
    val pointCount = distances.pointCount
    // Both scans of a round cost O(n) per point.
    val slices     = DriverParallelism.sliceRangesForWork(pointCount, pointCount.toLong)
    val medoids    = new Array[Int](k)
    val isSelected = new Array[Boolean](pointCount)

    medoids(0) = mostCentralPoint(distances, weights, slices)
    isSelected(medoids(0)) = true

    // Distance from each point to its nearest selected medoid.
    val distanceToNearestMedoid = Array.tabulate(pointCount)(j => distances(medoids(0), j))

    var selectedCount = 1
    while (selectedCount < k) {
      val next = largestGainPoint(distances, weights, distanceToNearestMedoid, isSelected, slices)
      medoids(selectedCount) = next
      isSelected(next) = true
      selectedCount += 1

      var j = 0
      while (j < pointCount) {
        val d = distances(next, j)
        if (d < distanceToNearestMedoid(j)) distanceToNearestMedoid(j) = d
        j += 1
      }
    }
    medoids
  }

  /** First medoid: the point minimising the total weighted distance to all others. */
  private def mostCentralPoint(
    distances: DistanceMatrix,
    weights:   Array[Double],
    slices:    IndexedSeq[(Int, Int)]
  ): Int = {
    val pointCount = distances.pointCount
    slices.par.map { case (from, until) =>
      var bestPoint = -1
      var bestTotal = Double.MaxValue
      var point     = from
      while (point < until) {
        var total = 0.0
        var j     = 0
        while (j < pointCount) { total += weights(j) * distances(point, j); j += 1 }
        if (total < bestTotal) { bestTotal = total; bestPoint = point }
        point += 1
      }
      (bestTotal, bestPoint)
    }.seq.reduce(keepSmallerTotal)._2
  }

  /** Next medoid: the unselected point maximising the weighted cost reduction it brings. */
  private def largestGainPoint(
    distances:               DistanceMatrix,
    weights:                 Array[Double],
    distanceToNearestMedoid: Array[Double],
    isSelected:              Array[Boolean],
    slices:                  IndexedSeq[(Int, Int)]
  ): Int = {
    val pointCount = distances.pointCount
    slices.par.map { case (from, until) =>
      var bestPoint = -1
      var bestGain  = Double.NegativeInfinity
      var candidate = from
      while (candidate < until) {
        if (!isSelected(candidate)) {
          var gain = 0.0
          var j    = 0
          while (j < pointCount) {
            val reduction = distanceToNearestMedoid(j) - distances(candidate, j)
            if (reduction > 0.0) gain += weights(j) * reduction
            j += 1
          }
          if (gain > bestGain) { bestGain = gain; bestPoint = candidate }
        }
        candidate += 1
      }
      (bestGain, bestPoint)
    }.seq.reduce(keepLargerGain)._2
  }

  /** Slices reduce in ascending order and the left one wins a tie, so the winner is the
   *  lowest index — the serial tie rule. An empty slice (all points selected) never wins. */
  private def keepSmallerTotal(a: (Double, Int), b: (Double, Int)): (Double, Int) =
    if (b._2 < 0 || a._1 <= b._1 && a._2 >= 0) a else b

  private def keepLargerGain(a: (Double, Int), b: (Double, Int)): (Double, Int) =
    if (b._2 < 0 || a._1 >= b._1 && a._2 >= 0) a else b
}
