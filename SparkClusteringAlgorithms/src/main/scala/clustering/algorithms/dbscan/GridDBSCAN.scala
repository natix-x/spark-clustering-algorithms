package clustering.algorithms.dbscan

import clustering.core.Clusterer
import clustering.data.{DatasetOps, Point}
import clustering.distance.{DistanceMetric, EuclideanDistance}
import clustering.utils.{SparkUtils, UnionFind}
import org.apache.spark.rdd.RDD


/** Grid-accelerated DBSCAN.
 *
 *  Cell side = eps → checking ±1 neighbours in each dimension is sufficient
 *  to find all eps-neighbours (cells differing by 2+ are always > eps apart).
 *  Reduces neighbourhood join from O(n²) to O(n * 3^d).
 */
class GridDBSCAN(
  val eps: Double,
  val minPts: Int,
  val distance: DistanceMetric = EuclideanDistance
) extends Clusterer {

  override def fit(data: RDD[Point]): DBSCANModel = {
    val cached = DatasetOps.cachePoints(data)
    implicit val sc = cached.sparkContext

    val cellPoints: RDD[(Vector[Int], Point)] = cached
      .map { p =>
        val cell = p.values.map(v => math.floor(v / eps).toInt).toVector
        (cell, p)
      }
      .cache()

    val neighborPairs: RDD[(Point, Point)] = cellPoints
      .flatMap { case (cell, p) => neighborCells(cell).map(nc => (nc, p)) }
      .join(cellPoints)
      .values
      .filter { case (p, q) => distance.compute(p, q) <= eps }
      .cache()

    val corePointsArray: Array[Point] = neighborPairs
      .map { case (p, _) => (p, 1) }
      .reduceByKey(_ + _)
      .filter { case (_, cnt) => cnt >= minPts }
      .keys
      .collect()

    val corePointsSet = corePointsArray.toSet
    val bcCorePoints  = SparkUtils.broadcastSafe(corePointsSet)

    val edges: Array[(Point, Point)] = neighborPairs
      .filter { case (p, q) =>
        val cp = bcCorePoints.value
        cp.contains(p) && cp.contains(q)
      }
      .collect()

    neighborPairs.unpersist(blocking = false)
    cellPoints.unpersist(blocking = false)
    bcCorePoints.unpersist(blocking = false)

    val coreLabels   = UnionFind.labelComponents(corePointsArray, edges)
    val bcCoreLabels = SparkUtils.broadcastSafe(coreLabels)

    val labelsRDD: RDD[(Point, Int)] = cached.mapPartitions { iter =>
      val localLabels = bcCoreLabels.value
      val localCores  = localLabels.keys.toArray
      iter.map { p =>
        if (localLabels.contains(p)) (p, localLabels(p))
        else {
          var i = 0; var label = -1; var found = false
          while (i < localCores.length && !found) {
            if (distance.compute(localCores(i), p) <= eps) {
              label = localLabels(localCores(i)); found = true
            }
            i += 1
          }
          (p, label)
        }
      }
    }

    new DBSCANModel(labelsRDD, corePointsSet, eps, distance)
  }

  private def neighborCells(cell: Vector[Int]): Seq[Vector[Int]] = {
    val offsets = Seq(-1, 0, 1)

    def expand(dims: List[Int], current: Vector[Int]): Seq[Vector[Int]] =
      dims match {
        case Nil       => Seq(current)
        case d :: rest => offsets.flatMap(o => expand(rest, current :+ (cell(d) + o)))
      }

    expand(cell.indices.toList, Vector.empty)
  }
}
