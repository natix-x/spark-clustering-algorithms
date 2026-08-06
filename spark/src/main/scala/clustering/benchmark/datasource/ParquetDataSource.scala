package clustering.benchmark.datasource

import clustering.benchmark.config.Params
import clustering.core.Columns
import org.apache.spark.ml.functions.array_to_vector
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.sql.{Column, DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, expr}
import org.apache.spark.sql.types.{ArrayType, NumericType}
import org.json4s._

final class ParquetDataSource(
  val path: String,
  val featureColumnName: String,
  val targetPartitionCount: Option[Int] = None,
  val weightColumnName: Option[String] = None,
  val sampleFraction: Option[Double] = None,
  val sampleSeed: Long = 42L
) extends DataSource {

  override val name: String = "parquet"

  override def metadata: Map[String, String] = Map(
    "format" -> "parquet",
    "path" -> path,
    "featureColumnName" -> featureColumnName,
    "sampleFraction" -> sampleFraction.map(_.toString).getOrElse("none"),
    "seed" -> sampleSeed.toString,
    "numPartitions" -> targetPartitionCount.map(_.toString).getOrElse("none"),
    "weightColumn" -> weightColumnName.getOrElse("none")
  )

  override def load(spark: SparkSession): DataFrame = {
    val allRows = spark.read.parquet(path)

    // Reduce rows before the projection, so both stay pushable into the scan.
    val sampledRows = sampleFraction.filter(_ < 1.0).map(allRows.sample(false, _, sampleSeed)).getOrElse(allRows)

    val weightProjection: Seq[Column] = weightColumnName.toSeq.map(name => col(name).cast("double").as(Columns.Weight))

    val projected = sampledRows.schema(featureColumnName).dataType match {
      case VectorType =>
        sampledRows.select(col(featureColumnName).as(Columns.Features) +: weightProjection: _*)
      case ArrayType(_: NumericType, _) =>
        // The embedding sets are arrays of float, so cast elements before packing.
        sampledRows.select(
          array_to_vector(expr(s"transform(`$featureColumnName`, x -> CAST(x AS DOUBLE))"))
            .as(Columns.Features) +: weightProjection: _*
        )
      case unsupportedType =>
        throw new IllegalArgumentException(
          s"ParquetDataSource: unsupported feature-column type for '$featureColumnName': $unsupportedType"
        )
    }

    // The read splits the files by size, which for a small dataset leaves most cores idle;
    // coalesce down / shuffle up to the run's partition count instead.
    targetPartitionCount match {
      case Some(target) =>
        val actual = projected.rdd.getNumPartitions
        if (target == actual) projected
        else if (target < actual) projected.coalesce(target)
        else projected.repartition(target)
      case None => projected
    }
  }
}

object ParquetDataSource {

  private implicit val formats: Formats = DefaultFormats

  object Factory extends DataSource.Factory {
    val typeName: String = "parquet"

    def create(params: JObject): DataSource = {
      val config = Params(params)
      new ParquetDataSource(
        path = config.string("path"),
        featureColumnName = config.string("featureColumnName"),
        targetPartitionCount = config.opt("numPartitions").map(_.extract[Int]),
        weightColumnName = config.opt("weightColumn").map(_.extract[String]),
        sampleFraction = config.opt("sampleFraction").map(_.extract[Double]),
        sampleSeed = config.longOpt("seed", 42L)
      )
    }
  }
}