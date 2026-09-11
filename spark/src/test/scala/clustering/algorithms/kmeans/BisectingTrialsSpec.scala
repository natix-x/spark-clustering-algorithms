package clustering.algorithms.kmeans

import clustering.algorithms.kmeans.hierarchical.{BisectingKMeans, BisectingKMeansModel}
import clustering.core.Columns
import clustering.distance.EuclideanDistance
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

import scala.util.Random

/** `trials` is Steinbach, Karypis & Kumar's ITER — the step that separates the implemented
 *  algorithm from the published one. What the knob has to guarantee:
 *
 *   1. `trials = 1` is the previous behaviour, bit for bit. It is the default, so every
 *      existing config and every number already measured must keep its meaning.
 *   2. more trials never pick a worse split, because the winner is chosen by the same cost
 *      the tree is built to minimise. This is what makes a `trials` sweep readable as a
 *      cost-vs-quality curve rather than as noise.
 *   3. a fixed `trials`'s COST is reproducible, so the variance a sweep reports is the
 *      platform's, not the knob's — checked by cost, not raw labels, since an exact tie between
 *      two candidates can legitimately flip which one wins under task-completion-order variance
 *      (Lloyd's fold no longer pins merge order, 11.09.2026) without the tree costing any more.
 */
class BisectingTrialsSpec extends AnyFunSuite with BeforeAndAfterAll {

  private var spark: SparkSession = _

  override def beforeAll(): Unit = {
    spark = SparkSession.builder()
      .master("local[2]")
      .appName("bisecting-trials-spec")
      .config("spark.ui.enabled", "false")
      .config("spark.sql.shuffle.partitions", "4")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
  }

  override def afterAll(): Unit = if (spark != null) spark.stop()

  /** Six well-separated blobs: enough structure that a bad first bisection is recoverable by
   *  a later trial, so the trials actually disagree instead of all landing on the same split. */
  private def blobs: DataFrame = {
    val rng     = new Random(7L)
    val centres = Seq((0.0, 0.0), (12.0, 0.0), (0.0, 12.0), (12.0, 12.0), (24.0, 6.0), (6.0, 24.0))
    val rows    = centres.flatMap { case (cx, cy) =>
      Seq.fill(60)(Vectors.dense(cx + rng.nextGaussian(), cy + rng.nextGaussian()): Vector)
    }
    val session = spark
    import session.implicits._
    rows.map(v => v -> 1.0).toDF(Columns.Features, Columns.Weight).repartition(4).cache()
  }

  /** Total weighted SSE of the fitted leaves — the objective the tree greedily minimises,
   *  measured against the leaf a point is actually routed to (root-to-leaf traversal), not
   *  against its globally nearest centroid. */
  private def costOf(model: BisectingKMeansModel, data: DataFrame): Double = {
    val centroids = model.clusterCentroids
    model.assignClusters(data)
      .select(Columns.Features, Columns.Prediction)
      .collect()
      .map { r =>
        val d = EuclideanDistance.compute(r.getAs[Vector](0), centroids(r.getInt(1)))
        d * d
      }
      .sum
  }

  private def modelCost(data: DataFrame, trials: Int, k: Int = 6): Double =
    costOf(new BisectingKMeans(k = k, seed = 11L, trials = trials).fit(data), data)

  test("trials = 1 leaves the single-shot bisection untouched") {
    val data = blobs
    // The default must BE the old path, not merely resemble it: same seed derivation, no
    // extra scoring job, therefore the same cost. Compared by cost, not raw labels: two
    // independent fits' centroid sums can legitimately differ in the last bit (fold no longer
    // pins merge order, 11.09.2026) without the underlying model actually differing.
    val viaDefault  = costOf(new BisectingKMeans(k = 6, seed = 11L).fit(data), data)
    val viaExplicit = modelCost(data, trials = 1)
    assert(math.abs(viaDefault - viaExplicit) < 1e-9, s"$viaDefault vs $viaExplicit")
  }

  test("more trials never yield a costlier SPLIT") {
    val data = blobs
    // k = 2 is the only k at which the tree IS one split, so the local guarantee is directly
    // observable: the winner is chosen by exactly the cost measured here, therefore competing
    // trials cannot lose to a single one.
    val one  = modelCost(data, trials = 1, k = 2)
    val five = modelCost(data, trials = 5, k = 2)
    assert(five <= one + 1e-9, s"trials=5 split cost $five exceeded trials=1 cost $one")
  }

  test("more trials CAN yield a costlier tree — greedy selection is not monotone") {
    val data = blobs
    val one  = modelCost(data, trials = 1)
    val five = modelCost(data, trials = 5)
    // Recorded deliberately, because it contradicts the intuition the knob invites. `trials`
    // optimises ONE bisection at a time, but the tree is built greedily: the leaf split next
    // is whichever currently costs most, so a locally better split changes which leaf is
    // picked in every later round. A better first cut can therefore route the search into a
    // worse final tree, and here it does — 6 blobs, seed 11: 753.5 at trials = 1 against 871.0
    // at trials = 5. So `trials` buys per-split quality and stability, NOT a monotone
    // improvement in the final objective, and a sweep must be read that way.
    assert(five > one, "expected the documented anomaly; re-check the note above if it vanished")
  }

  test("a fixed trials count is reproducible") {
    val data = blobs
    // Cost, not raw labels: an exact tie between two of the `trials` candidates can legitimately
    // flip which one wins under task-completion-order variance (fold no longer pins merge order,
    // 11.09.2026) without the resulting tree costing any more — and cost is what this file's other
    // sweep comparisons need reproducible, not which candidate broke a tie.
    val a = modelCost(data, trials = 4)
    val b = modelCost(data, trials = 4)
    assert(math.abs(a - b) < 1e-9, s"$a vs $b")
  }

  test("trials must be at least one") {
    val data = blobs
    val thrown = intercept[IllegalArgumentException](new BisectingKMeans(k = 3, trials = 0).fit(data))
    assert(thrown.getMessage.contains("trials"))
  }
}
