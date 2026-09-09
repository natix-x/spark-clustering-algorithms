package clustering.algorithms.kmedoids

import clustering.algorithms.kmedoids.distributed.DistributedFastPAM
import clustering.algorithms.kmedoids.hybrid.PAMAE
import clustering.algorithms.kmedoids.components.{MedoidCost, MedoidRefinement}
import clustering.core.{Columns, Weights}
import clustering.distance.EuclideanDistance
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

/** Weight invariance for the DISTRIBUTED medoid entries. Lives in this package so the shared
 *  primitive ([[MedoidRefinement]]) can be driven directly.
 *
 *  Why not compare medoid coordinates: duplicating a row makes each copy a separate candidate,
 *  so a solver may fill two slots with two copies of one coordinate — a tied optimum, not a
 *  different one. The invariant is therefore equality of the OBJECTIVE, and for the refinement
 *  primitive it is checked with an identical init and pool, which removes the last source of
 *  divergence (a uniform pool sample necessarily differs between 6 weighted and 16 duplicated
 *  rows).
 */
class WeightedMedoidsSpec extends AnyFunSuite with BeforeAndAfterAll {

  private var spark: SparkSession = _

  override def beforeAll(): Unit = {
    spark = SparkSession.builder()
      .master("local[2]")
      .appName("weighted-medoids-spec")
      .config("spark.ui.enabled", "false")
      .config("spark.sql.shuffle.partitions", "4")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
  }

  override def afterAll(): Unit = if (spark != null) spark.stop()

  private val groups: Seq[(Vector, Int)] = Seq(
    Vectors.dense(0.0, 0.0)  -> 1,
    Vectors.dense(0.4, 0.0)  -> 1,
    Vectors.dense(10.0, 0.0) -> 5,
    Vectors.dense(10.4, 0.0) -> 5,
    Vectors.dense(30.0, 0.0) -> 2,
    Vectors.dense(30.4, 0.0) -> 2
  )

  private def weighted: DataFrame = {
    val session = spark
    import session.implicits._
    groups.map { case (v, w) => (v, w.toDouble) }.toDF(Columns.Features, Columns.Weight).repartition(3)
  }

  private def duplicated: DataFrame = {
    val session = spark
    import session.implicits._
    groups.flatMap { case (v, w) => Seq.fill(w)(v) }.map(Tuple1.apply).toDF(Columns.Features).repartition(3)
  }

  private def objective(medoids: Array[Vector]): Double =
    groups.map { case (v, w) => w * medoids.map(m => EuclideanDistance.compute(v, m)).min }.sum

  test("the refinement primitive is weight-invariant given the same init and pool") {
    val pool    = groups.map(_._1).toArray
    val initial = pool.take(3)

    val w = MedoidRefinement.refine(Weights.toRdd(weighted),   initial, pool, EuclideanDistance, 10)
    val d = MedoidRefinement.refine(Weights.toRdd(duplicated), initial, pool, EuclideanDistance, 10)

    assert(w.medoids.map(_.toString).toSeq == d.medoids.map(_.toString).toSeq,
      s"weighted=${w.medoids.mkString(",")} duplicated=${d.medoids.mkString(",")}")
    assert(math.abs(w.cost - d.cost) < 1e-9, s"cost ${w.cost} != ${d.cost}")
  }

  test("MedoidCost agrees between the weighted and duplicated forms") {
    val medoids = Array(groups(1)._1, groups(2)._1, groups(4)._1)
    val onWeighted   = MedoidCost.total(weighted, medoids, EuclideanDistance)
    val onDuplicated = MedoidCost.total(duplicated, medoids, EuclideanDistance)
    assert(math.abs(onWeighted - onDuplicated) < 1e-9)
    assert(math.abs(onWeighted - objective(medoids)) < 1e-9)
  }

  test("distfastpam and pamae reach the same objective on both forms") {
    def dist(df: DataFrame)  = new DistributedFastPAM(3, 50, EuclideanDistance).fit(df).medoids
    def pamae(df: DataFrame) = new PAMAE(3, 2, 6, 50, 3, 6, "fastpam", EuclideanDistance, 3L).fit(df).medoids

    Seq("distfastpam" -> (dist(weighted), dist(duplicated)),
        "pamae"       -> (pamae(weighted), pamae(duplicated))).foreach { case (name, (w, d)) =>
      assert(math.abs(objective(w) - objective(d)) < 1e-9,
        s"$name: ${objective(w)} != ${objective(d)}")
    }
  }

  /** Every row of `duplicated` is repeated, so a solver that treats each copy as a separate
   *  candidate could fill two slots with the same coordinate and leave a cluster empty. */
  test("pamae returns k DISTINCT medoids on data whose every row is repeated") {
    val medoids = new PAMAE(3, 2, 6, 50, 3, 6, "fastpam", EuclideanDistance, 3L).fit(duplicated).medoids
    assert(medoids.map(_.toString).distinct.length == 3, s"duplicate medoids: ${medoids.mkString(",")}")
  }
}
