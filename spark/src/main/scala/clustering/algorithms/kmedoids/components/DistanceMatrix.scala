package clustering.algorithms.kmedoids.components

import clustering.algorithms.kmedoids.local.{FastPAM, FasterPAM, PAM}
import clustering.distance.DistanceMetric
import clustering.utils.DriverParallelism
import org.apache.spark.ml.linalg.Vector

/** Dense pairwise distances in one flat row-major array — the shared state of the driver-local
 *  rungs ([[PAM]], [[FastPAM]], [[FasterPAM]]). See `docs/kmedoids_docs.md`. */
private[kmedoids] final class DistanceMatrix private (values: Array[Double], val pointCount: Int) {

  def apply(from: Int, to: Int): Double = values(from * pointCount + to)
}

private[kmedoids] object DistanceMatrix {

  /** Only the upper triangle is computed; rows are filled in parallel on the driver. Entry
   *  (i, j) with i < j and its mirror (j, i) are both written by row i alone, so nothing races. */
  def pairwise(points: Array[Vector], distance: DistanceMetric): DistanceMatrix = {
    val pointCount = points.length
    val values     = new Array[Double](pointCount * pointCount)

    // Row i does n − i computations, so slices are STRIDED — contiguous ranges would leave the
    // last slice nearly idle.
    val totalPairs = pointCount.toLong * pointCount / 2
    DriverParallelism.strideSlicesForWork(pointCount, totalPairs).par.foreach { rows =>
      rows.foreach { row =>
        var column = row + 1
        while (column < pointCount) {
          val d = distance.compute(points(row), points(column))
          values(row * pointCount + column) = d
          values(column * pointCount + row) = d
          column += 1
        }
      }
    }
    new DistanceMatrix(values, pointCount)
  }
}
