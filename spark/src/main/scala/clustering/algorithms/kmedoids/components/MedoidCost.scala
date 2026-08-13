package clustering.algorithms.kmedoids.components

import clustering.algorithms.kmedoids.hybrid.CLARA
import clustering.core.{Columns, Weights}
import clustering.distance.DistanceMetric
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
    data:       DataFrame,
    medoidSets: Array[Array[Vector]],
    distance:   DistanceMetric
  ): Array[Double] = {
    require(medoidSets.nonEmpty, "perMedoidSet: at least one medoid set is required")

    val broadcastSets = data.sparkSession.sparkContext.broadcast(medoidSets)
    val metric        = distance

    val costColumns = medoidSets.indices.map { setIndex =>
      val nearestDistance = udf { features: Vector =>
        val medoids = broadcastSets.value(setIndex)
        var nearest = Double.MaxValue
        var slot    = 0
        while (slot < medoids.length) {
          val d = metric.compute(features, medoids(slot))
          if (d < nearest) nearest = d
          slot += 1
        }
        nearest
      }
      sum(nearestDistance(col(Columns.Features)) * col(Columns.Weight)).as(s"cost$setIndex")
    }

    val row = Weights.withWeights(data).agg(costColumns.head, costColumns.tail: _*).head()
    broadcastSets.unpersist(blocking = false)

    // `sum` over an empty dataset is null.
    Array.tabulate(medoidSets.length)(i => if (row.isNullAt(i)) 0.0 else row.getDouble(i))
  }
}
