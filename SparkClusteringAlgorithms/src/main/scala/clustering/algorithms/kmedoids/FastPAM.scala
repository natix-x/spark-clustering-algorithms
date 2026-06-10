package clustering.algorithms.kmedoids

import clustering.core.Clusterer
import clustering.data.{DatasetOps, Point}
import clustering.distance.{DistanceMetric, EuclideanDistance}
import org.apache.spark.rdd.RDD

class FastPAM(
  val k: Int,
  val maxIter: Int = 100,
  val distance: DistanceMetric = EuclideanDistance
) extends Clusterer {

  override def fit(data: RDD[Point]): KMedoidsModel = {
    // UWAGA: collect() pobiera całe dane na driver.
    // Optymalne tylko dla zbiorów mieszczących się w pamięci RAM drivera.
    val points = DatasetOps.cachePoints(data).collect()
    val n      = points.length

    // 1D Array dla lepszego cache locality w JVM
    val dist   = precomputeDistances(points, n)

    var medoids  = buildPhase(dist, n)
    var iter     = 0
    var improved = true

    // Tablice pomocnicze reużywane w każdej iteracji, aby unikać alokacji pamięci
    val isMedoid = new Array[Boolean](n)
    val d1 = new Array[Double](n)
    val d2 = new Array[Double](n)
    val nearestMedoidIdx = new Array[Int](n)

    while (improved && iter < maxIter) {
      val (nextMedoids, didImprove) = swapPhaseFast(dist, n, medoids, isMedoid, d1, d2, nearestMedoidIdx)
      medoids  = nextMedoids
      improved = didImprove
      iter    += 1
    }

    new KMedoidsModel(medoids.map(points), distance)
  }

  // Używamy płaskiej tablicy 1D: dist(i * n + j) zamiast dist(i)(j)
  private def precomputeDistances(points: Array[Point], n: Int): Array[Double] = {
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

  private def buildPhase(dist: Array[Double], n: Int): Array[Int] = {
    val selected = new Array[Int](k)
    var selCount = 0

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
    selCount += 1

    val dBest = new Array[Double](n)
    var j = 0
    while (j < n) {
      dBest(j) = dist(bestFirst * n + j)
      j += 1
    }

    while (selCount < k) {
      var bestH = -1
      var bestGain = Double.MinValue

      var h = 0
      while (h < n) {
        // Sprawdzamy czy h już nie jest wybrane (proste wyszukiwanie linowe jest szybkie dla małego k)
        var alreadySelected = false
        var c = 0
        while (c < selCount && !alreadySelected) {
          if (selected(c) == h) alreadySelected = true
          c += 1
        }

        if (!alreadySelected) {
          var gain = 0.0
          var j = 0
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

      // Aktualizacja dBest dla następnej iteracji
      j = 0
      while (j < n) {
        val d = dist(bestH * n + j)
        if (d < dBest(j)) dBest(j) = d
        j += 1
      }
    }

    selected
  }

  private def swapPhaseFast(
    dist: Array[Double],
    n: Int,
    medoids: Array[Int],
    isMedoid: Array[Boolean],
    d1: Array[Double],
    d2: Array[Double],
    nearestMedoidIdx: Array[Int]
  ): (Array[Int], Boolean) = {

    // Resetowanie i wypełnianie flag isMedoid
    var i = 0
    while (i < n) { isMedoid(i) = false; i += 1 }
    var mi = 0
    while (mi < k) { isMedoid(medoids(mi)) = true; mi += 1 }

    // Prekalkulacja najbliższego (d1) i drugiego najbliższego (d2) medoidu dla każdego punktu
    var j = 0
    while (j < n) {
      var min1 = Double.MaxValue
      var min2 = Double.MaxValue
      var m1Idx = -1

      var m = 0
      while (m < k) {
        val d = dist(j * n + medoids(m))
        if (d < min1) {
          min2 = min1
          min1 = d
          m1Idx = m
        } else if (d < min2) {
          min2 = d
        }
        m += 1
      }
      d1(j) = min1
      d2(j) = min2
      nearestMedoidIdx(j) = m1Idx
      j += 1
    }

    var bestSwapMi = -1
    var bestSwapH = -1
    var bestDelta = 0.0

    // Szukamy najlepszej pary do zamiany (medoid -> nowy punkt)
    var h = 0
    while (h < n) {
      if (!isMedoid(h)) {
        mi = 0
        while (mi < k) {
          var currentDelta = 0.0
          j = 0
          while (j < n) {
            val dh = dist(j * n + h)
            if (nearestMedoidIdx(j) == mi) {
              // Punkt j traci swój główny medoid, przechodzi na d2(j) lub dh
              currentDelta += (math.min(dh, d2(j)) - d1(j))
            } else {
              // Punkt j zachowuje swój główny medoid, ale może zyskać lepszy dh
              if (dh < d1(j)) {
                currentDelta += (dh - d1(j))
              }
            }
            j += 1
          }

          if (currentDelta < bestDelta) {
            bestDelta = currentDelta
            bestSwapMi = mi
            bestSwapH = h
          }
          mi += 1
        }
      }
      h += 1
    }

    // Jeśli znaleźliśmy zamianę, która redukuje całkowity koszt
    if (bestDelta < 0.0) {
      val newMedoids = medoids.clone()
      newMedoids(bestSwapMi) = bestSwapH
      (newMedoids, true)
    } else {
      (medoids, false)
    }
  }
}