package clustering.algorithms.kmedoids

import clustering.core.Clusterer
import clustering.distance.{DistanceMetric, EuclideanDistance}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col

/** Partitioning Around Medoids (PAM) — classic k-medoids clustering algorithm.
 *
 *  PAM is more robust to noise and outliers than K-Means because it uses
 *  actual data points (medoids) as cluster centres instead of arbitrary centroids.
 *
 *  Scalability note: PAM is O(k*(n-k)²) per iteration and requires O(n²) memory
 *  for the distance matrix. It is only feasible for small datasets (n ≤ ~10 000).
 *  For large datasets use CLARA, which runs PAM on small random samples.
 *
 *  Two entry points:
 *    fit(data: DataFrame) — collects the `features` column to the driver and runs PAM locally.
 *    fitLocal(points: Array[Vector]) — runs PAM on an already-collected array;
 *      used by CLARA to avoid the DataFrame → collect() round-trip.
 */
class PAM(
  val k:        Int,
  val maxIter:  Int          = 100,
  val distance: DistanceMetric = EuclideanDistance
) extends Clusterer {

  /** DataFrame entry point — collects to driver and delegates to fitLocal.
   *  Only suitable for datasets that fit comfortably in driver memory.
   */
  override def fit(data: DataFrame): KMedoidsModel =
    fitLocal(data.select(col("features")).collect().map(_.getAs[Vector]("features")))

  /** Local entry point used by CLARA — no Spark overhead.
   *  All computation happens on the driver using flat arrays.
   */
  def fitLocal(points: Array[Vector]): KMedoidsModel = {
    val n = points.length

    require(n >= k, s"Dataset too small: n=$n points but k=$k medoids requested.")

    // Step 1: Precompute all pairwise distances into a flat 1D array.
    val dist = precomputeDistances(points, n)

    // Step 2: BUILD phase — greedy initial medoid selection.
    var medoids  = buildPhase(dist, n)
    var iter     = 0
    var improved = true

    // Step 3: SWAP phase — iteratively improve until convergence or maxIter.
    while (improved && iter < maxIter) {
      val (nextMedoids, didImprove) = swapPhaseStandard(dist, n, medoids)
      medoids  = nextMedoids
      improved = didImprove
      iter    += 1
    }

    new KMedoidsModel(medoids.map(points), distance)
  }

  // ── Private helpers ────────────────────────────────────────────────────────

  /** Computes a pairwise distance matrix stored in a flat 1D array.
   *
   *  Flat layout (dist(i*n + j)) improves CPU cache locality and avoids
   *  JVM pointer-chasing overhead of 2D arrays.
   *  Symmetry is exploited: only the upper triangle is computed.
   */
  private def precomputeDistances(points: Array[Vector], n: Int): Array[Double] = {
    val dist = new Array[Double](n * n)
    var i = 0
    while (i < n) {
      var j = i + 1
      while (j < n) {
        val d = distance.compute(points(i), points(j))
        dist(i * n + j) = d
        dist(j * n + i) = d
        j += 1
      }
      i += 1
    }
    dist
  }

  /** BUILD phase: greedy selection of k initial medoids.
   *
   *  1. First medoid: the point minimising total distance to all others.
   *  2. Remaining k-1 medoids: each chosen to maximise incremental cost reduction.
   */
  private def buildPhase(dist: Array[Double], n: Int): Array[Int] = {
    val selected = new Array[Int](k)
    var selCount = 0

    // Step A: first medoid — closest to the dataset's centre of mass.
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
    selCount   += 1

    // dBest(j) = distance from point j to its nearest selected medoid.
    val dBest = new Array[Double](n)
    var j = 0
    while (j < n) { dBest(j) = dist(bestFirst * n + j); j += 1 }

    // Step B: greedily add remaining k-1 medoids.
    while (selCount < k) {
      var bestH    = -1
      var bestGain = Double.MinValue

      var h = 0
      while (h < n) {
        var alreadySelected = false
        var c = 0
        while (c < selCount && !alreadySelected) {
          if (selected(c) == h) alreadySelected = true
          c += 1
        }

        if (!alreadySelected) {
          var gain = 0.0
          var j    = 0
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

      j = 0
      while (j < n) {
        val d = dist(bestH * n + j)
        if (d < dBest(j)) dBest(j) = d
        j += 1
      }
    }

    selected
  }

  /** Standard SWAP phase: O(k*(n-k)²) exhaustive search.
   *
   *  Tests every (medoid, non-medoid) swap and applies the one with the
   *  greatest cost reduction. Returns (newMedoids, improved).
   */
  private def swapPhaseStandard(
    dist:    Array[Double],
    n:       Int,
    medoids: Array[Int]
  ): (Array[Int], Boolean) = {

    var bestSwapMi = -1
    var bestSwapH  = -1
    var bestDelta  = 0.0

    val candidateMedoids = medoids.clone()

    var mi = 0
    while (mi < k) {
      val oldMedoid = medoids(mi)

      var h = 0
      while (h < n) {
        var isAlreadyMedoid = false
        var mIdx = 0
        while (mIdx < k) {
          if (medoids(mIdx) == h) isAlreadyMedoid = true
          mIdx += 1
        }

        if (!isAlreadyMedoid) {
          candidateMedoids(mi) = h

          var currentDelta = 0.0
          var j = 0
          while (j < n) {
            var oldMin = Double.MaxValue
            var m = 0
            while (m < k) {
              val d = dist(j * n + medoids(m))
              if (d < oldMin) oldMin = d
              m += 1
            }

            var newMin = Double.MaxValue
            m = 0
            while (m < k) {
              val d = dist(j * n + candidateMedoids(m))
              if (d < newMin) newMin = d
              m += 1
            }

            currentDelta += (newMin - oldMin)
            j += 1
          }

          if (currentDelta < bestDelta) {
            bestDelta  = currentDelta
            bestSwapMi = mi
            bestSwapH  = h
          }

          candidateMedoids(mi) = oldMedoid
        }
        h += 1
      }
      mi += 1
    }

    if (bestDelta < 0.0) {
      val newMedoids = medoids.clone()
      newMedoids(bestSwapMi) = bestSwapH
      (newMedoids, true)
    } else {
      (medoids, false)
    }
  }
}
