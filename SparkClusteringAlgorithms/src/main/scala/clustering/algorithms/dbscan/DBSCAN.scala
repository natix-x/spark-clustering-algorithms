package clustering.algorithms.dbscan

import clustering.core.Clusterer
import clustering.data.{DatasetOps, Point}
import clustering.distance.{DistanceMetric, EuclideanDistance}
import clustering.utils.{SparkUtils, UnionFind}
import org.apache.spark.rdd.RDD


/** Distributed DBSCAN using a full cartesian neighbourhood join.
 *
 *  Complexity: O(n²) due to cartesian product — intentionally naive to
 *  serve as a scalability baseline against GridDBSCAN.
 *
 *  Algorithm:
 *  1. Cartesian join → all (p, q) pairs within eps.
 *  2. Count neighbours per point → identify core points.
 *  3. Build edge set among core points → union-find on driver.
 *  4. Assign border points to nearest core cluster; mark rest as noise.
 */
class DBSCAN(
  val eps: Double,
  val minPts: Int,
  val distance: DistanceMetric = new EuclideanDistance()
) extends Clusterer {

  override def fit(data: RDD[Point]): DBSCANModel = {
    val cached = DatasetOps.cachePoints(data)
    implicit val sc = cached.sparkContext

    val withinEps: RDD[(Point, Point)] = cached
      .cartesian(cached)
      .filter { case (p, q) => distance.compute(p, q) <= eps }
      .cache()

    val corePoints: Set[Point] = withinEps
      .map { case (p, _) => (p, 1) }
      .reduceByKey(_ + _)
      .filter { case (_, cnt) => cnt >= minPts }
      .keys
      .collect()
      .toSet

    val bcCorePoints = SparkUtils.broadcastSafe(corePoints)

    val edges: Array[(Point, Point)] = withinEps
      .filter { case (p, q) =>
        val cp = bcCorePoints.value
        cp.contains(p) && cp.contains(q)
      }
      .collect()

    withinEps.unpersist()
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
}
