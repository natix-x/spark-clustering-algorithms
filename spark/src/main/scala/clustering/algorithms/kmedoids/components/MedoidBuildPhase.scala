package clustering.algorithms.kmedoids.components

/** BUILD phase (Kaufman & Rousseeuw 1990): greedy selection of k initial medoids, shared by every
 *  driver-local rung so they differ only in their SWAP strategy.
 *
 *  Both scans of every round run over the point range in ascending order, so a tie is resolved in
 *  favour of the lowest index — see `docs/kmedoids_docs.md`.
 */
private[kmedoids] object MedoidBuildPhase {

  /** @param weights `weights(j)` = how many points `j` stands for
   *  @return indices of the k initial medoids, in selection order */
  def selectInitialMedoids(distances: DistanceMatrix, k: Int, weights: Array[Double]): Array[Int] = {
    val pointCount = distances.pointCount
    val medoids = new Array[Int](k)
    val isSelected = new Array[Boolean](pointCount)

    medoids(0) = mostCentralPoint(distances, weights)
    isSelected(medoids(0)) = true

    val distanceToNearestMedoid = Array.tabulate(pointCount)(j => distances(medoids(0), j))

    var selectedCount = 1
    while (selectedCount < k) {
      val next = largestGainPoint(distances, weights, distanceToNearestMedoid, isSelected)
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
    weights: Array[Double]
  ): Int = {
    val pointCount = distances.pointCount
    var bestPoint = -1
    var bestTotal = Double.MaxValue
    var point = 0
    while (point < pointCount) {
      var total = 0.0
      var j = 0
      while (j < pointCount) { total += weights(j) * distances(point, j); j += 1 }
      if (total < bestTotal) { bestTotal = total; bestPoint = point }
      point += 1
    }
    bestPoint
  }

  /** Next medoid: the unselected point maximising the weighted cost reduction it brings. */
  private def largestGainPoint(
    distances: DistanceMatrix,
    weights: Array[Double],
    distanceToNearestMedoid: Array[Double],
    isSelected: Array[Boolean]
  ): Int = {
    val pointCount = distances.pointCount
    var bestPoint = -1
    var bestGain = Double.NegativeInfinity
    var candidate = 0
    while (candidate < pointCount) {
      if (!isSelected(candidate)) {
        var gain = 0.0
        var j = 0
        while (j < pointCount) {
          val reduction = distanceToNearestMedoid(j) - distances(candidate, j)
          if (reduction > 0.0) gain += weights(j) * reduction
          j += 1
        }
        if (gain > bestGain) { bestGain = gain; bestPoint = candidate }
      }
      candidate += 1
    }
    bestPoint
  }
}
