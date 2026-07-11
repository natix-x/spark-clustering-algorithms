package clustering.benchmark.datasource

import clustering.benchmark.config.Params
import clustering.core.Columns
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.types.{StructField, StructType}
import org.json4s._

import scala.util.Random

/** Generates the 5-mode 3D mixture used as a stress test for the algorithms.
 *
 *  Five modes — noise + two overlapping Gaussians + an elongated mode + a dense
 *  distant mode — parameterized by config so a single matrix entry can scale n
 *  by orders of magnitude without recompiling.
 *
 *  Params (all optional except `numPoints`):
 *    - numPoints: Long
 *    - numPartitions: Int (default 8)
 *    - seed: Long (default 42)
 */
final class SyntheticDataSource(
  val numPoints: Long,
  val numPartitions: Int,
  val seed: Long
) extends DataSource {

  require(numPoints >= 0L, s"SyntheticDataSource: numPoints must be >= 0, got $numPoints")
  require(numPartitions > 0, s"SyntheticDataSource: numPartitions must be > 0, got $numPartitions")

  override val name: String = "synthetic-5mix-3d"

  override def metadata: Map[String, String] = Map(
    "format"        -> "synthetic",
    "generator"     -> name,
    "numPoints"     -> numPoints.toString,
    "numPartitions" -> numPartitions.toString,
    "seed"          -> seed.toString,
    "nFeatures"     -> "3"
  )

  override def load(spark: SparkSession): DataFrame = {
    val sc        = spark.sparkContext
    val localSeed = seed
    val rowRdd = sc.parallelize(0L until numPoints, numPartitions)
      .mapPartitionsWithIndex { (partIdx, iter) =>
        // Per-partition deterministic seeding so results are reproducible
        // across runs without serializing a single mutable Random.
        val rng = new Random(localSeed + partIdx * 0x9E3779B97F4A7C15L)
        iter.map(_ => Row(SyntheticDataSource.samplePoint(rng)))
      }
    val schema = StructType(Seq(StructField(Columns.Features, VectorType, nullable = false)))
    spark.createDataFrame(rowRdd, schema)
  }
}

object SyntheticDataSource {

  object Factory extends DataSource.Factory {
    val typeName: String = "synthetic"

    def create(params: JObject): DataSource = {
      val p = Params(params)
      new SyntheticDataSource(
        numPoints     = p.long("numPoints"),
        numPartitions = p.intOpt("numPartitions", 8),
        seed          = p.longOpt("seed", 42L)
      )
    }
  }

  /** One sample from the 5-mode mixture:
   *  noise / twin-A / twin-B / cigar / micro-dense. */
  private def samplePoint(rng: Random): Vector = {
    val r = rng.nextDouble()
    if (r < 0.05) {
      // Extreme outliers in [-1000, 1000]^3
      Vectors.dense(
        rng.nextDouble() * 2000 - 1000,
        rng.nextDouble() * 2000 - 1000,
        rng.nextDouble() * 2000 - 1000
      )
    } else if (r < 0.30) {
      // Twin A around (0, 0, 0)
      Vectors.dense(rng.nextGaussian() * 5, rng.nextGaussian() * 5, rng.nextGaussian() * 5)
    } else if (r < 0.55) {
      // Twin B around (5, 5, 5) — overlapping with A
      Vectors.dense(rng.nextGaussian() * 5 + 5, rng.nextGaussian() * 5 + 5, rng.nextGaussian() * 5 + 5)
    } else if (r < 0.70) {
      // Elongated mode around (100, 50, 50), high variance on X
      Vectors.dense(
        rng.nextGaussian() * 40 + 100,
        rng.nextGaussian() * 2  + 50,
        rng.nextGaussian() * 2  + 50
      )
    } else {
      // Dense distant mode around (200, 200, 200)
      Vectors.dense(
        rng.nextGaussian() * 1 + 200,
        rng.nextGaussian() * 1 + 200,
        rng.nextGaussian() * 1 + 200
      )
    }
  }
}
