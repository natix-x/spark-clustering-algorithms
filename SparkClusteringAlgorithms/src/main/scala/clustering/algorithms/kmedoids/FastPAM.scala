package clustering.algorithms.kmedoids

import clustering.core.Clusterer
import clustering.distance.{DistanceMetric, EuclideanDistance}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col

/**
 * FastPAM - An optimized, high-performance variant of the Partitioning Around Medoids algorithm.
 *
 * FastPAM reduces the complexity of the SWAP phase from O(k^2 * n^2) to O(k * n^2) by
 * caching the nearest (d1) and second-nearest (d2) medoid distances for every point.
 * This avoids full re-evaluation of the entire cluster configuration during every swap test.
 */
class FastPAM(
  val k: Int,
  val maxIter: Int = 100,
  val distance: DistanceMetric = EuclideanDistance
) extends Clusterer {

  override def fit(data: DataFrame): KMedoidsModel = {
    // Note: collect() pulls all distributed data into the driver memory.
    // This implementation assumes the dataset fits on a single machine.
    val points = data.select(col("features")).collect().map(_.getAs[Vector]("features"))
    val n      = points.length

    // Step 1: Precompute all pairwise distances into a flat 1D array for CPU cache locality
    val dist   = precomputeDistances(points, n)

    // Step 2: BUILD phase - Greedy heuristic initialization
    var medoids  = buildPhase(dist, n)
    var iter     = 0
    var improved = true

    // Step 3: Pre-allocate state arrays to prevent GC overhead and memory allocations inside the loop
    val isMedoid         = new Array[Boolean](n)
    val d1               = new Array[Double](n)
    val d2               = new Array[Double](n)
    val nearestMedoidIdx = new Array[Int](n)

    // Step 4: Iterative optimization using FastPAM swap logic
    while (improved && iter < maxIter) {
      val (nextMedoids, didImprove) = swapPhaseFast(dist, n, medoids, isMedoid, d1, d2, nearestMedoidIdx)
      medoids  = nextMedoids
      improved = didImprove
      iter    += 1
    }

    new KMedoidsModel(medoids.map(points), distance)
  }

  /**
   * Computes pairwise distances into a 1D flat array to ensure maximum spatial locality.
   * Row-major index formula: dist(i * n + j) maps to matrix element [i][j].
   */
  private def precomputeDistances(points: Array[Vector], n: Int): Array[Double] = {
    val dist = new Array[Double](n * n)
    var i = 0
    while (i < n) {
      var j = i + 1
      while (j < n) {
        val d = distance.compute(points(i), points(j))
        dist(i * n + j) = d
        dist(j * n + i) = d // Distance symmetry
        j += 1
      }
      i += 1
    }
    dist
  }

  /**
   * BUILD Phase: Greedily selects initial medoids to minimize global distance.
   */
  private def buildPhase(dist: Array[Double], n: Int): Array[Int] = {
    val selected = new Array[Int](k)
    var selCount = 0

    // --- Step A: Find the first medoid (minimizes total distance to all other points) ---
    var bestFirst = 0
    var bestSum   = Double.MaxValue
    var i = 0
    while (i < n) {
      var s = 0.0
      var j = 0
      while (j < n) { s += dist(i * n + j); j += 1 }
      if (s < bestSum) { bestSum = s; bestFirst = i }
      i += 1
    }

    selected(0) = bestFirst
    selCount += 1

    // Cache to track minimum distance from each point to any currently chosen medoid
    val dBest = new Array[Double](n)
    var j = 0
    while (j < n) {
      dBest(j) = dist(bestFirst * n + j)
      j += 1
    }

    // --- Step B: Successively select remaining k-1 medoids ---
    while (selCount < k) {
      var bestH = -1
      var bestGain = Double.MinValue

      var h = 0
      while (h < n) {
        // Linear scan to check if point 'h' is already a medoid (highly efficient for small k)
        var alreadySelected = false
        var c = 0
        while (c < selCount && !alreadySelected) {
          if (selected(c) == h) alreadySelected = true
          c += 1
        }

        if (!alreadySelected) {
          // Compute cumulative distance reduction (gain) if h is added as a medoid
          var gain = 0.0
          var j = 0
          while (j < n) {
            val g = dBest(j) - dist(h * n + j)
            if (g > 0.0) gain += g
            j += 1
          }
          if (gain > bestGain) { bestGain = gain; bestH = h }
        }
        h += 1
      }
      selected(selCount) = bestH
      selCount += 1

      // Update the dBest distance cache incorporating the new medoid
      j = 0
      while (j < n) {
        val d = dist(bestH * n + j)
        if (d < dBest(j)) dBest(j) = d
        j += 1
      }
    }

    selected
  }

  /**
   * FastPAM SWAP Phase: Evaluates potential swaps in O(k * n^2) time complexity.
   *
   * Instead of recomputing configuration costs from scratch, this method determines
   * the exact change in cost (Delta) instantly based on whether a point loses its closest
   * medoid or finds a closer alternative.
   */
  private def swapPhaseFast(
    dist: Array[Double],
    n: Int,
    medoids: Array[Int],
    isMedoid: Array[Boolean],
    d1: Array[Double],
    d2: Array[Double],
    nearestMedoidIdx: Array[Int]
  ): (Array[Int], Boolean) = {

    // Reset and populate the fast medoid-lookup boolean array
    var i = 0
    while (i < n) { isMedoid(i) = false; i += 1 }
    var mi = 0
    while (mi < k) { isMedoid(medoids(mi)) = true; mi += 1 }

    // Precalculate d1 (closest), d2 (second closest), and nearestMedoidIdx for all data points
    var j = 0
    while (j < n) {
      var min1 = Double.MaxValue
      var min2 = Double.MaxValue
      var m1Idx = -1

      var m = 0
      while (m < k) {
        val d = dist(j * n + medoids(m))
        if (d < min1) {
          min2 = min1
          min1 = d
          m1Idx = m // Store index within the medoids array [0, k-1]
        } else if (d < min2) {
          min2 = d
        }
        m += 1
      }
      d1(j) = min1
      d2(j) = min2
      nearestMedoidIdx(j) = m1Idx
      j += 1
    }

    var bestSwapMi = -1
    var bestSwapH = -1
    var bestDelta = 0.0 // Look for the most negative Delta (maximum savings)

    // Evaluate every non-medoid point 'h' as a replacement candidate
    var h = 0
    while (h < n) {
      if (!isMedoid(h)) {

        // Evaluate swapping out current medoid at index 'mi'
        mi = 0
        while (mi < k) {
          var currentDelta = 0.0
          j = 0
          while (j < n) {
            val dh = dist(j * n + h)

            if (nearestMedoidIdx(j) == mi) {
              // Case 1: Point j loses its primary medoid.
              // It must reassign to either the new candidate 'h' or its second-nearest medoid 'd2(j)'.
              currentDelta += (math.min(dh, d2(j)) - d1(j))
            } else {
              // Case 2: Point j keeps its primary medoid.
              // However, if the new candidate 'h' is closer than its primary medoid, it gains an advantage.
              if (dh < d1(j)) {
                currentDelta += (dh - d1(j))
              }
            }
            j += 1
          }

          // Track the absolute best swap configuration found across all passes
          if (currentDelta < bestDelta) {
            bestDelta = currentDelta
            bestSwapMi = mi
            bestSwapH = h
          }
          mi += 1
        }
      }
      h += 1
    }

    // Apply the swap if a cost reduction was confirmed
    if (bestDelta < 0.0) {
      val newMedoids = medoids.clone()
      newMedoids(bestSwapMi) = bestSwapH
      (newMedoids, true)
    } else {
      (medoids, false) // Local optimum achieved, stop swapping
    }
  }
}