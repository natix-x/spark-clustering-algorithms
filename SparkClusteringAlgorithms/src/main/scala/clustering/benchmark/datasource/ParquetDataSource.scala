package clustering.benchmark.datasource

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.functions.array_to_vector
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, expr}
import org.apache.spark.sql.types.ArrayType
import org.json4s._

/** Loads a `DataFrame` (single `features` Vector column) from a Parquet file.
 *
 *  Two input layouts are supported via config:
 *
 *  1. **Multi-column** — `featureColumns: ["x", "y", "z"]`. Each named column
 *     must be numeric (Int/Long/Float/Double). One row -> one Point.
 *
 *  2. **Single vector column** — `featureColumn: "features"` where the column
 *     holds an `Array[Double]` or Spark ML `Vector`. Useful when an upstream
 *     pipeline already produced embeddings (ImageNet features, TF-IDF...).
 *
 *  Exactly one of `featureColumns` / `featureColumn` must be present.
 *
 *  Optional params:
 *    - `path` (required): Parquet file or directory
 *    - `repartition` (optional Int): coalesce/repartition after read to vary
 *       the partition count without changing the file layout — useful for
 *       partition-strategy experiments.
 */
final class ParquetDataSource(
  val path:           String,
  val featureColumns: Option[Seq[String]],
  val featureColumn:  Option[String],
  val repartition:    Option[Int]
) extends DataSource {

  require(
    featureColumns.isDefined ^ featureColumn.isDefined,
    "ParquetDataSource: exactly one of featureColumns or featureColumn must be set"
  )

  override val name: String = "parquet"

  override def metadata: Map[String, String] = Map(
    "format"      -> "parquet",
    "path"        -> path,
    "layout"      -> (if (featureColumns.isDefined) "multi-column" else "vector-column"),
    "columns"     -> featureColumns.map(_.mkString(",")).orElse(featureColumn).getOrElse(""),
    "repartition" -> repartition.map(_.toString).getOrElse("none")
  )

  override def load(spark: SparkSession): DataFrame = {
    val df0 = spark.read.parquet(path)

    val features: DataFrame = featureColumns match {
      case Some(cols) =>
        // Multi-column layout: cast each to double and assemble into a Vector.
        val casted = df0.selectExpr(cols.map(c => s"CAST(`$c` AS DOUBLE) AS `$c`"): _*)
        new VectorAssembler()
          .setInputCols(cols.toArray)
          .setOutputCol("features")
          .transform(casted)
          .select(col("features"))

      case None =>
        // Single-column layout: already a Vector, or an Array of numbers.
        val c  = featureColumn.get
        val dt = df0.schema(c).dataType
        if (dt == VectorType) {
          df0.select(col(c).as("features"))
        } else dt match {
          case _: ArrayType =>
            // Cast array elements to double, then pack into a Vector.
            df0.select(
              array_to_vector(expr(s"transform(`$c`, x -> CAST(x AS DOUBLE))")).as("features")
            )
          case other =>
            throw new IllegalArgumentException(
              s"ParquetDataSource: unsupported feature-column type for '$c': $other"
            )
        }
    }

    repartition match {
      case Some(p) if p > 0 && p != features.rdd.getNumPartitions =>
        if (p < features.rdd.getNumPartitions) features.coalesce(p)
        else features.repartition(p)
      case _ => features
    }
  }
}

object ParquetDataSource {

  private implicit val formats: Formats = DefaultFormats

  object Factory extends DataSource.Factory {
    val typeName: String = "parquet"

    def create(params: JObject): DataSource = {
      val path = (params \ "path").extract[String]
      val featureColumns = (params \ "featureColumns") match {
        case JNothing => None
        case v        => Some(v.extract[Seq[String]])
      }
      val featureColumn = (params \ "featureColumn") match {
        case JNothing  => None
        case JString(s) => Some(s)
        case other     => throw new IllegalArgumentException(s"featureColumn must be a string, got $other")
      }
      val repartition = (params \ "repartition") match {
        case JNothing => None
        case v        => Some(v.extract[Int])
      }
      new ParquetDataSource(path, featureColumns, featureColumn, repartition)
    }
  }
}