package clustering.algorithms.kmedoids

import clustering.core.Clusterer
import clustering.data.{DatasetOps, Point}
import clustering.distance.{DistanceMetric, EuclideanDistance}
import org.apache.spark.rdd.RDD

/** Clustering Large Applications (CLARA).
 *
 * Runs PAM on `numSamples` random subsets of the data (each of fixed size
 * `sampleSize`), then scores every candidate model on the
 * full dataset and returns the best one.
 *
 * Scales to extremely large datasets where full PAM would be prohibitive.
 */
class CLARA(
             val k: Int,
             val numSamples: Int = 5,
             val sampleSize: Int = 1000, // 🔹 ZMIANA: Stały rozmiar zamiast ułamka!
             val maxIter: Int = 100,
             val distance: DistanceMetric = EuclideanDistance
           ) extends Clusterer {

  // Zwróć uwagę, że PAM będzie teraz działał na bardzo małych zbiorach (np. 1000 punktów).
  // Dzięki temu możemy go uruchamiać bardzo szybko w pętli.
  private val pam = new PAM(k, maxIter, distance)

  override def fit(data: RDD[Point]): KMedoidsModel = {
    val cached = DatasetOps.cachePoints(data)
    val sc = cached.sparkContext

    var bestModel: KMedoidsModel = null
    var minCost = Double.MaxValue

    // Iterujemy sekwencyjnie po próbkach
    for (i <- 0 until numSamples) {

      // 1. Zbieramy małą, stałą próbkę prosto na Drivera (tzw. "Driver-side PAM")
      // takeSample to akcja, więc dane od razu lądują w pamięci operacyjnej Mastera
      val sampleArray = cached.takeSample(withReplacement = false, num = sampleSize, seed = i.toLong)

      // Tworzymy lokalne RDD z próbki (lub wywołujemy PAM bezpośrednio, w zależności od tego jak masz napisany PAM)
      val sampleRDD = sc.parallelize(sampleArray, 2)

      // 2. Uruchamiamy klasyczny algorytm K-Medoids na maleńkiej próbce
      val candidateModel = pam.fit(sampleRDD)

      // 3. Oceniamy JAKOŚĆ tej próbki na wszystkich 50 milionach punktów (Rozproszone)
      val cost = evaluateCost(cached, candidateModel)

      if (cost < minCost) {
        minCost = cost
        bestModel = candidateModel
      }
    }

    bestModel
  }

  private def evaluateCost(data: RDD[Point], model: KMedoidsModel): Double = {
    implicit val sc = data.sparkContext
    val dist        = distance
    val bcMedoids   = sc.broadcast(model.medoids)

    val cost = data.mapPartitions { iter =>
      val localMedoids = bcMedoids.value
      var partSum = 0.0

      while (iter.hasNext) {
        val p = iter.next()
        var minD = Double.MaxValue
        var j    = 0
        while (j < localMedoids.length) {
          val d = dist.compute(localMedoids(j), p)
          if (d < minD) minD = d
          j += 1
        }
        partSum += minD
      }
      Iterator(partSum)
    }.sum()

    bcMedoids.unpersist(blocking = false) // 🔹 ZMIANA: Bezpieczne usuwanie z pamięci
    cost
  }
}