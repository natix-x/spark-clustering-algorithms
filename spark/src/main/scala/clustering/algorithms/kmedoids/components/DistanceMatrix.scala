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

  /** Only the upper triangle is computed; entry (i, j) with i < j and its mirror (j, i) are both
   *  written while visiting row i. */
  def pairwise(points: Array[Vector], distance: DistanceMetric): DistanceMatrix = {
    val pointCount = points.length
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
