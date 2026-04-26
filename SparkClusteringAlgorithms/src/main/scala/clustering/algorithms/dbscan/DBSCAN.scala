package clustering.algorithms.dbscan

import clustering.core.Clusterer
import clustering.data.{DatasetOps, Point}
import clustering.distance.{DistanceMetric, EuclideanDistance}
import clustering.utils.UnionFind
import org.apache.spark.rdd.RDD

class DBSCAN(
              val eps: Double,
              val minPts: Int,
              val distance: DistanceMetric = EuclideanDistance
            ) extends Clusterer {

  override def fit(data: RDD[Point]): DBSCANModel = {
    val cached = DatasetOps.cachePoints(data)
    implicit val sc = cached.sparkContext

    // 1. Kartezjańskie szukanie sąsiadów (O(N^2))
    val withinEps: RDD[(Point, Point)] = cached
      .cartesian(cached)
      .filter { case (p, q) => distance.compute(p, q) <= eps }
      .cache()

    // 2. Szukanie punktów rdzeniowych (Core Points)
    val corePointsArray: Array[Point] = withinEps
      .map { case (p, _) => (p, 1L) }
      .reduceByKey(_ + _)
      .filter { case (_, cnt) => cnt >= minPts }
      .keys
      .collect()

    val corePointsSet = corePointsArray.toSet
    val bcCorePoints = sc.broadcast(corePointsSet)

    // 3. Budowanie krawędzi i spójnych składowych
    val edges: Array[(Point, Point)] = withinEps
      .filter { case (p, q) =>
        val cp = bcCorePoints.value
        cp.contains(p) && cp.contains(q)
      }
      .collect()

    withinEps.unpersist(blocking = false)
    bcCorePoints.unpersist(blocking = false)

    val coreLabels: Map[Point, Int] = UnionFind.labelComponents(corePointsArray, edges)
    val bcCoreLabels = sc.broadcast(coreLabels)

    // 4. Przypisanie szumu i punktów brzegowych (W PEŁNI ROZPROSZONE)
    val labeledRDD: RDD[(Point, Int)] = cached.mapPartitions { iter =>
      val localCoreLabelsMap = bcCoreLabels.value
      val localCorePointsArr = localCoreLabelsMap.keys.toArray

      iter.map { p =>
        if (localCoreLabelsMap.contains(p)) {
          (p, localCoreLabelsMap(p))
        } else {
          var i = 0
          var assignedLabel = -1 // -1 = NOISE
          var found = false
          val len = localCorePointsArr.length

          while (i < len && !found) {
            val cp = localCorePointsArr(i)
            if (distance.compute(cp, p) <= eps) {
              assignedLabel = localCoreLabelsMap(cp)
              found = true
            }
            i += 1
          }
          (p, assignedLabel)
        }
      }
    }

    // 🔹 KLUCZOWA ZMIANA: Zwracamy RDD, NIE ROBIMY collectAsMap()!
    new DBSCANModel(labeledRDD, corePointsSet, eps, distance)
  }
}