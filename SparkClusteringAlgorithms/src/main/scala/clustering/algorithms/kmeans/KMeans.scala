package clustering.algorithms.kmeans

import clustering.core.Clusterer
import clustering.data.{DatasetOps, Point}
import clustering.distance.{DistanceMetric, EuclideanDistance}
import clustering.utils.Convergence
import org.apache.spark.rdd.RDD

class KMeans(
              val k: Int,
              val maxIter: Int = 100,
              val eps: Double = 1e-4,
              val distance: DistanceMetric = EuclideanDistance
            ) extends Clusterer {

  override def fit(data: RDD[Point]): KMeansModel = {
    // Zakładam, że cachePoints robi data.cache() pod spodem
    val cached = DatasetOps.cachePoints(data)
    implicit val sc = cached.sparkContext

    var centroids = cached.takeSample(withReplacement = false, num = k, seed = 42L)
    var iter      = 0
    var converged = false

    while (!converged && iter < maxIter) {
      val bcCentroids = sc.broadcast(centroids)

      // 1. MAP + LOKALNA OPTYMALIZACJA IMPERATYWNA
      val mappedToCentroids = cached.mapPartitions { iterator =>
        val localCentroids = bcCentroids.value

        iterator.map { p =>
          var bestIdx = 0
          var minD = Double.MaxValue
          var j = 0

          // Eliminacja MathUtils.argmin na rzecz bezalokacyjnej pętli
          while (j < localCentroids.length) {
            val d = distance.compute(p, localCentroids(j))
            if (d < minD) {
              minD = d
              bestIdx = j
            }
            j += 1
          }

          // Emitujemy parę: (Klucz=ID klastra, Wartość=(Punkt, Licznik=1L))
          (bestIdx, (p, 1L))
        }
      }

      // 2. REDUCE BY KEY (Rozwiązanie problemu Shuffle / OOM)
      // Zamiast groupByKey, agregujemy sumy lokalnie na węzłach.
      // Wymaga to, aby Twoja klasa Point umiała dodać do siebie dwa punkty (p1 + p2)
      val newCentroidSums = mappedToCentroids.reduceByKey { case ((p1, count1), (p2, count2)) =>
        // Założenie: Point ma metodę dodawania (lub należy to zrobić ręcznie na tablicach)
        (p1 + p2, count1 + count2)
      }.collectAsMap()

      // 3. AKTUALIZACJA CENTROIDÓW
      // Zwalnianie asynchroniczne zamiast agresywnego .destroy()
      bcCentroids.unpersist(blocking = false)

      val newCentroids = (0 until k).map { i =>
        newCentroidSums.get(i) match {
          case Some((sumPoint, count)) =>
            // Dzielimy sumę wektorów przez liczbę punktów.
            // Założenie: Point ma operator dzielenia przez skalar (sumPoint / count)
            sumPoint / count
          case None =>
            // Jeśli klaster "umarł" (0 punktów), zostawiamy stary centroid
            centroids(i)
        }
      }.toArray

      // 4. SPRAWDZENIE ZBIEŻNOŚCI
      converged  = Convergence.hasConverged(centroids, newCentroids, eps, distance)
      centroids  = newCentroids
      iter      += 1
    }

    new KMeansModel(centroids, distance)
  }
}