package clustering.algorithms.kmedoids.components

import clustering.algorithms.kmedoids.local.{FastPAM, FasterPAM}
import clustering.distance.DistanceMetric
import org.apache.spark.ml.linalg.Vector

/** Dense pairwise distances in one flat row-major array — the shared state of the driver-local
 *  rungs ([[FastPAM]], [[FasterPAM]]). See `docs/kmedoids_docs.md`. */
private[kmedoids] final class DistanceMatrix private (values: Array[Double], val pointCount: Int) {

  def apply(from: Int, to: Int): Double = values(from * pointCount + to)
}

private[kmedoids] object DistanceMatrix {

  /** Share of the driver heap this matrix may occupy. */
  private val HeapBudgetFraction = 0.5

  /** Rejects sizes whose dense matrix cannot be allocated: past the JVM array-length cap, or past
   *  the heap budget. Fails here with an actionable message instead of deep inside the allocation. */
  private def requireFits(pointCount: Int): Unit = {
    val cellCount     = pointCount.toLong * pointCount.toLong
    val requiredBytes = cellCount * 8L
    val maxHeapBytes  = Runtime.getRuntime.maxMemory()
    val budgetBytes   = (maxHeapBytes * HeapBudgetFraction).toLong
    require(cellCount <= Int.MaxValue.toLong,
      s"FastPAM/FasterPAM's exact medoid search precomputes a dense $pointCount x $pointCount " +
      "pairwise-distance matrix on the driver as ONE array, whose length is capped at " +
      s"${Int.MaxValue} entries by the JVM regardless of available memory — this candidate/coreset " +
      "size cannot be handled by this exact, driver-local solver at all. Reduce the candidate size " +
      "(e.g. CLARA's sampleSize, coreset's m) or switch to inner: kmeans, which does not precompute " +
      "an O(n^2) distance matrix.")
    require(requiredBytes <= budgetBytes,
      s"FastPAM/FasterPAM's exact medoid search precomputes a dense $pointCount x $pointCount " +
      s"pairwise-distance matrix on the driver (${requiredBytes / (1024 * 1024)} MB) — this exceeds " +
      s"the driver's safe budget (${budgetBytes / (1024 * 1024)} MB of ${maxHeapBytes / (1024 * 1024)} " +
      "MB heap). Reduce the candidate/coreset size (e.g. CLARA's sampleSize, coreset's m) or switch " +
      "to inner: kmeans, which does not precompute an O(n^2) distance matrix.")
  }

  /** Only the upper triangle is computed; entry (i, j) with i < j and its mirror (j, i) are both
   *  written while visiting row i. */
  def pairwise(points: Array[Vector], distance: DistanceMetric): DistanceMatrix = {
    val pointCount = points.length
    requireFits(pointCount)
    val values = new Array[Double](pointCount * pointCount)
    // Unpacked ONCE per point, not once per pair: this loop runs n²/2 times, so a wrapper
    // dereference and a dense/sparse type match inside the metric are paid n²/2 times too.
    val coords = points.map(_.toArray)

    var row = 0
    while (row < pointCount) {
      var column = row + 1
      while (column < pointCount) {
        val d = distance.compute(coords(row), coords(column))
        values(row * pointCount + column) = d
        values(column * pointCount + row) = d
        column += 1
      }
      row += 1
    }
    new DistanceMatrix(values, pointCount)
  }
}
