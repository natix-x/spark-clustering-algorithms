package clustering.algorithms.kmedoids

import clustering.core.Clusterer
import clustering.distance.{DistanceMetric, EuclideanDistance}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col

/** Distributed FastPAM — k-medoids that spreads the O(n²) work across the CLUSTER instead
 *  of collecting everything and building an O(n²) distance matrix on the driver (as the
 *  plain [[FastPAM]] does). The distance matrix is NEVER materialised: the needed pairwise
 *  distances are recomputed on the fly, partitioned across executors.
 *
 *  Mirror of the Flink `DistributedFastPAM` so the two frameworks run the SAME algorithm.
 *
 *  ==What is distributed==
 *  The candidate coordinates `C` (all n points, O(n·dim)) are collected once and broadcast;
 *  the points `j` stay in the partitioned RDD. Each partition folds its LOCAL points into
 *  per-candidate partial sums; the driver sums the partials (sorted by partition id for
 *  determinism) and picks the best move. The Σ-over-j that drives BUILD and SWAP is thus
 *  split by executor; the driver only ever holds O(n) coordinates and an O(n·k) result.
 *
 *  ==FastPAM1 SWAP, distributed==
 *  For the current medoid set, swapping medoid-slot `i` for non-medoid candidate `h` changes
 *  the cost by Δ(h,i) = shared(h) + removeLoss(h)(i) (Schubert & Rousseeuw, FastPAM1), where
 *  over all points j with nearest/second distances d1,d2 and nearest slot n1:
 *    shared(h)         = Σ_j [ d(h,j) < d1(j) ? d(h,j) − d1(j) : 0 ]
 *    removeLoss(h)(i)  = Σ_{j: n1(j)=i} [ (min(d2(j), d(h,j)) − d1(j)) − sharedContrib_j(h) ]
 *  One Spark job per BUILD step (k jobs) and per SWAP round (≤ maxIter jobs).
 *
 *  ==Determinism==
 *  Partials are summed in partition-id order, so results are reproducible for a FIXED
 *  partition count. Across different partitionings the floating-point summation order
 *  changes and near-tie moves may resolve differently — exact bit-identity across cluster
 *  sizes is not achievable for a distributed sum. */
class DistributedFastPAM(
  val k:        Int,
  val maxIter:  Int            = 100,
  val distance: DistanceMetric = EuclideanDistance
) extends Clusterer {

  override def fit(data: DataFrame): KMedoidsModel = {
    val sc = data.sparkSession.sparkContext

    val points: RDD[Vector] = data.select(col("features")).rdd
      .map(_.getAs[Vector]("features")).cache()

    // Candidate coordinates: collected once (O(n·dim)), broadcast to every executor.
    // This is NOT the O(n²) matrix — that is never built.
    val c: Array[Vector] = points.collect()
    val n = c.length
    require(n >= k, s"Dataset too small: n=$n points but k=$k medoids requested.")
    val bcC = sc.broadcast(c)

    val medoids  = buildPhase(points, bcC, n)
    val isMedoid = Array.fill(n)(false)
    medoids.foreach(idx => isMedoid(idx) = true)

    var iter     = 0
    var improved = true
    while (improved && iter < maxIter) {
      improved = swapRound(points, bcC, n, medoids, isMedoid)
      iter += 1
    }

    bcC.destroy()
    points.unpersist(blocking = false)
    new KMedoidsModel(medoids.map(c), distance)
  }

  // ── BUILD ──────────────────────────────────────────────────────────────────

  /** Greedy BUILD, one distributed job per medoid. */
  private def buildPhase(points: RDD[Vector], bcC: Broadcast[Array[Vector]], n: Int): Array[Int] = {
    val sc       = points.sparkContext
    val medoids  = new Array[Int](k)
    val taken    = Array.fill(n)(false)
    val dist     = distance

    // First medoid: minimise Σ_j d(h, j).
    val totalDist = aggregateArray(points, n) { (acc, x) =>
      val cc = bcC.value
      var h = 0
      while (h < n) { acc(h) += dist.compute(cc(h), x); h += 1 }
    }
    val first = argMinExcluding(totalDist, taken)
    medoids(0) = first
    taken(first) = true

    // Remaining k-1: maximise Σ_j max(0, dBest(j) − d(h, j)).
    var t = 1
    while (t < k) {
      val selected = medoids.take(t).map(bcC.value)
      val bcSel    = sc.broadcast(selected)
      val gain = aggregateArray(points, n) { (acc, x) =>
        val cc = bcC.value
        val ss = bcSel.value
        var dBest = Double.MaxValue
        var s = 0
        while (s < ss.length) { val d = dist.compute(ss(s), x); if (d < dBest) dBest = d; s += 1 }
        var h = 0
        while (h < n) { val g = dBest - dist.compute(cc(h), x); if (g > 0.0) acc(h) += g; h += 1 }
      }
      bcSel.destroy()
      val best = argMaxExcluding(gain, taken)
      medoids(t) = best
      taken(best) = true
      t += 1
    }
    medoids
  }

  // ── SWAP ─────────────────────────────────────────────────────────────────

  /** One distributed FastPAM1 SWAP round. Mutates `medoids`/`isMedoid` in place when an
   *  improving swap is found; returns whether one was applied. */
  private def swapRound(
    points:   RDD[Vector],
    bcC:      Broadcast[Array[Vector]],
    n:        Int,
    medoids:  Array[Int],
    isMedoid: Array[Boolean]
  ): Boolean = {
    val sc   = points.sparkContext
    val dist = distance
    val kk   = k
    val medoidCoords = medoids.map(bcC.value)
    val bcM  = sc.broadcast(medoidCoords)

    // flat(0 until n)        = shared(h)
    // flat(n + h*k + i)      = removeLoss(h)(i)
    val flat = aggregateArray(points, n + n * kk) { (acc, x) =>
      val cc = bcC.value
      val mm = bcM.value
      var d1 = Double.MaxValue
      var d2 = Double.MaxValue
      var n1 = -1
      var m = 0
      while (m < kk) {
        val d = dist.compute(mm(m), x)
        if (d < d1) { d2 = d1; d1 = d; n1 = m } else if (d < d2) { d2 = d }
        m += 1
      }
      var h = 0
      while (h < n) {
        val dj            = dist.compute(cc(h), x)
        val sharedContrib = if (dj < d1) dj - d1 else 0.0
        acc(h) += sharedContrib
        val caseB = math.min(d2, dj) - d1
        acc(n + h * kk + n1) += (caseB - sharedContrib)
        h += 1
      }
    }
    bcM.destroy()

    // Pick the best swap on the driver. Sequential FastPAM tie order: ascending h, then
    // ascending i, strict improvement.
    var bestDelta = 0.0
    var bestH     = -1
    var bestI     = -1
    var h = 0
    while (h < n) {
      if (!isMedoid(h)) {
        val shared = flat(h)
        val base   = n + h * kk
        var i = 0
        while (i < kk) {
          val delta = shared + flat(base + i)
          if (delta < bestDelta) { bestDelta = delta; bestH = h; bestI = i }
          i += 1
        }
      }
      h += 1
    }

    if (bestDelta < 0.0) {
      isMedoid(medoids(bestI)) = false
      medoids(bestI) = bestH
      isMedoid(bestH) = true
      true
    } else {
      false
    }
  }

  // ── Distributed aggregation plumbing ────────────────────────────────────────

  /** Folds each partition's local points into an `Array[Double]` of length `len`, then
   *  sums the per-partition arrays on the driver in partition-id order (determinism). */
  private def aggregateArray(points: RDD[Vector], len: Int)(fold: (Array[Double], Vector) => Unit): Array[Double] = {
    val partials: Array[(Int, Array[Double])] = points.mapPartitionsWithIndex { (idx, it) =>
      val acc = new Array[Double](len)
      it.foreach(x => fold(acc, x))
      Iterator.single((idx, acc))
    }.collect()

    val sum = new Array[Double](len)
    partials.sortBy(_._1).foreach { case (_, a) =>
      var i = 0
      while (i < len) { sum(i) += a(i); i += 1 }
    }
    sum
  }

  private def argMinExcluding(v: Array[Double], excluded: Array[Boolean]): Int = {
    var best = -1
    var min  = Double.MaxValue
    var i = 0
    while (i < v.length) {
      if (!excluded(i) && v(i) < min) { min = v(i); best = i }
      i += 1
    }
    best
  }

  private def argMaxExcluding(v: Array[Double], excluded: Array[Boolean]): Int = {
    var best = -1
    var max  = Double.NegativeInfinity
    var i = 0
    while (i < v.length) {
      if (!excluded(i) && v(i) > max) { max = v(i); best = i }
      i += 1
    }
    best
  }
}