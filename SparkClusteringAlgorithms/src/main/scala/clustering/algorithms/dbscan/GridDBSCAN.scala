package clustering.algorithms.dbscan

import clustering.core.Clusterer
import clustering.data.{DatasetOps, Point}
import clustering.distance.{DistanceMetric, EuclideanDistance}
import clustering.utils.{SparkUtils, UnionFind}
import org.apache.spark.rdd.RDD


/** Grid-accelerated DBSCAN.
 *
 *  Partitions the feature space into a regular grid with cell side
 *  `eps / sqrt(d)`.  For each point only the 3^d adjacent cells need
 *  to be checked, reducing the neighbourhood join from O(n²) to
 *  O(n * 3^d).  Particularly effective for low-dimensional data.
 *
 *  Algorithm:
 *  1. Assign every point to its grid cell.
 *  2. Expand each point to all neighbouring cells (flatMap).
 *  3. Join with the cell→point index → candidate pairs.
 *  4. Filter candidates by true eps-distance → neighbour pairs.
 *  5. Count neighbours → core points → union-find on driver.
 *  6. Assign border/noise labels.
 */
class GridDBSCAN(
  val eps: Double,
  val minPts: Int,
  val distance: DistanceMetric = new EuclideanDistance()
) extends Clusterer {

  override def fit(data: RDD[Point]): DBSCANModel = {
    val cached = DatasetOps.cachePoints(data)
    implicit val sc = cached.sparkContext

    val cellSize = eps

    val cellPoints: RDD[(Vector[Int], Point)] = cached
      .map { p =>
        val cell = p.values.map(v => math.floor(v / cellSize).toInt).toVector
        (cell, p)
      }
      .cache()

    val neighborPairs: RDD[(Point, Point)] = cellPoints
      .flatMap { case (cell, p) => neighborCells(cell).map(nc => (nc, p)) }
      .join(cellPoints)
      .values
      .filter { case (p, q) => distance.compute(p, q) <= eps }
      .cache()

    val corePoints: Set[Point] = neighborPairs
      .map { case (p, _) => (p, 1) }
      .reduceByKey(_ + _)
      .filter { case (_, cnt) => cnt >= minPts }
      .keys
      .collect()
      .toSet

    val bcCorePoints = SparkUtils.broadcastSafe(corePoints)

    val edges: Array[(Point, Point)] = neighborPairs
      .filter { case (p, q) =>
        val cp = bcCorePoints.value
        cp.contains(p) && cp.contains(q)
      }
      .collect()

    neighborPairs.unpersist()
    cellPoints.unpersist()
    bcCorePoints.destroy()

    val coreLabels: Map[Point, Int] = UnionFind.labelComponents(corePoints.toArray, edges)

    val allLabels: Map[Point, Int] = cached.collect().map { p =>
      if (corePoints.contains(p)) p -> coreLabels(p)
      else {
        val nearest = corePoints.find(cp => distance.compute(cp, p) <= eps)
        p -> nearest.map(coreLabels).getOrElse(-1)
      }
    }.toMap

    new DBSCANModel(allLabels, corePoints, eps, distance)
  }

  private def neighborCells(cell: Vector[Int]): Seq[Vector[Int]] = {
    val offsets = Seq(-1, 0, 1)

    def expand(dims: List[Int], current: Vector[Int]): Seq[Vector[Int]] =
      dims match {
        case Nil         => Seq(current)
        case d :: rest   => offsets.flatMap(o => expand(rest, current :+ (cell(d) + o)))
      }

    expand(cell.indices.toList, Vector.empty)
  }
}
