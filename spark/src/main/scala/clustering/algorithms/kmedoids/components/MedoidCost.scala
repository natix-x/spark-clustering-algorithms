package clustering.algorithms.kmedoids.components

import clustering.algorithms.kmedoids.hybrid.CLARA
import clustering.core.Weights
import clustering.distance.DistanceMetric
import clustering.utils.PartitionAggregator
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

/** The k-medoids objective on the FULL dataset: Σ_x w_x · min_m d(x, m) — the quantity every
 *  sampling method needs but none optimises directly. One broadcast + one aggregation (P1). */
private[kmedoids] object MedoidCost {

  def total(data: DataFrame, medoids: Array[Vector], distance: DistanceMetric): Double =
    perMedoidSet(data, Array(medoids), distance).head

  /** Cost of SEVERAL medoid sets in ONE pass — one `sum` aggregate per set over the same scan, so
   *  [[CLARA]] scores all its candidates with a single job. */
  def perMedoidSet(
    data: DataFrame,
    medoidSets: Array[Array[Vector]],
    distance: DistanceMetric
  ): Array[Double] = {
    require(medoidSets.nonEmpty, "perMedoidSet: at least one medoid set is required")

    // Broadcast raw coordinates: unpacked once here, so the per-row scan never dereferences a
    // wrapper, and `distanceUpTo` against the running minimum lets each candidate's coordinate
    // loop quit as soon as it cannot win — the Euclidean metric then pays `sqrt` per HIT, not
    // per comparison.
    val broadcastSets = data.sparkSession.sparkContext.broadcast(medoidSets.map(_.map(_.toArray)))
    val metric = distance

    // ONE fold: S set costs at once, coordinates unpacked once and shared
    //
    // Unordered: feeds only CLARA's argmin over candidate sets — a discrete pick, same category
    // as distfastpam/dbscanpp, so bit-order doesn't matter.
    val costs = PartitionAggregator.aggregateDoubles(
      Weights.toRdd(data), medoidSets.length) { (acc, row) =>
        val point = row._1.toArray
        val w     = row._2
        val sets  = broadcastSets.value
        var setIndex = 0
        while (setIndex < sets.length) {
          val medoids = sets(setIndex)
          var nearest = Double.MaxValue
          var slot = 0
          while (slot < medoids.length) {
            // distanceUpTo against the running minimum: each candidate's coordinate loop quits as
            // soon as it cannot win, so Euclidean pays `sqrt` per HIT and not per comparison.
            val d = metric.distanceUpTo(point, medoids(slot), nearest)
            if (d < nearest) nearest = d
            slot += 1
          }
          acc(setIndex) += w * nearest
          setIndex += 1
        }
    }

    broadcastSets.unpersist(blocking = false)
    costs
  }
}
