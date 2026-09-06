package clustering.algorithms.kmedoids.components

import clustering.algorithms.kmedoids.hybrid.CLARA
import clustering.core.{Columns, Weights}
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

    // ONE ordered fold over the data, accumulating all S set costs at once. The DataFrame form
    // this replaces built S separate UDF columns, so every row paid the VectorUDT round-trip and
    // `toArray` S TIMES — the same per-row overhead Lloyd's rewrite removed (measured 5.3× there,
    // 5.09.2026), multiplied by the number of candidate sets. Here the coordinates are unpacked
    // once and the S scans share them.
    //
    // Ordered merge, S doubles: these costs pick CLARA's winning sample, so two runs of one
    // configuration must not disagree in the last bits and then choose differently.
    val costs = PartitionAggregator.aggregateDoublesOrdered(
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
