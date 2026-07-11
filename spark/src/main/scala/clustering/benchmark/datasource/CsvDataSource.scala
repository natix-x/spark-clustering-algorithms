package clustering.benchmark.datasource

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.col
import org.json4s._

/** Loads a `DataFrame` (single `features` Vector column) from a CSV file (or directory of CSVs).
 *
 *  Required params:
 *    - `path`: file or directory
 *    - `featureColumns`: column names (when header=true) or indices as strings
 *      (when header=false)
 *
 *  Optional params:
 *    - `header`: true/false (default true)
 *    - `delimiter`: e.g. "," / ";" / "\t" (default ",")
 *    - `repartition`: change partition count after load
 *    - `quote`, `escape`: for non-standard CSVs (default `"` / `\`)
 */
final class CsvDataSource(
  val path:           String,
  val featureColumns: Seq[String],
  val header:         Boolean,
  val delimiter:      String,
  val quote:          String,
  val escape:         String,
  val repartition:    Option[Int]
) extends DataSource {

  override val name: String = "csv"

  override def metadata: Map[String, String] = Map(
    "format"      -> "csv",
    "path"        -> path,
    "header"      -> header.toString,
    "delimiter"   -> delimiter,
    "columns"     -> featureColumns.mkString(","),
    "nFeatures"   -> featureColumns.length.toString,
    "repartition" -> repartition.map(_.toString).getOrElse("none")
  )

  override def load(spark: SparkSession): DataFrame = {
    val reader = spark.read
      .option("header",    header.toString)
      .option("delimiter", delimiter)
      .option("quote",     quote)
      .option("escape",    escape)
      .option("inferSchema", "false") // we cast explicitly below for predictable types

    val df0 = reader.csv(path)

    // When header is false, Spark assigns _c0, _c1, ... — caller is expected
    // to pass those as featureColumns (e.g. ["_c0", "_c1", "_c2"]).
    val castExprs = featureColumns.map(c => s"CAST(`$c` AS DOUBLE) AS `$c`")
    val casted    = df0.selectExpr(castExprs: _*)

    val assembled = new VectorAssembler()
      .setInputCols(featureColumns.toArray)
      .setOutputCol("features")
      .transform(casted)
      .select(col("features"))

    repartition match {
      case Some(p) if p > 0 && p != assembled.rdd.getNumPartitions =>
        if (p < assembled.rdd.getNumPartitions) assembled.coalesce(p)
        else assembled.repartition(p)
      case _ => assembled
    }
  }
}

object CsvDataSource {

  private implicit val formats: Formats = DefaultFormats

  object Factory extends DataSource.Factory {
    val typeName: String = "csv"

    def create(params: JObject): DataSource = {
      val path           = (params \ "path").extract[String]
      val featureColumns = (params \ "featureColumns").extract[Seq[String]]
      require(featureColumns.nonEmpty, "CsvDataSource: featureColumns must not be empty")
      val header    = optBool(params, "header", default = true)
      val delimiter = optStr(params, "delimiter", default = ",")
      val quote     = optStr(params, "quote", default = "\"")
      val escape    = optStr(params, "escape", default = "\\")
      val repartition = (params \ "repartition") match {
        case JNothing => None
        case v        => Some(v.extract[Int])
      }
      new CsvDataSource(path, featureColumns, header, delimiter, quote, escape, repartition)
    }

    private def optBool(p: JObject, key: String, default: Boolean): Boolean =
      (p \ key) match {
        case JNothing  => default
        case JBool(b)  => b
        case other     => throw new IllegalArgumentException(s"$key must be boolean, got $other")
      }

    private def optStr(p: JObject, key: String, default: String): String =
      (p \ key) match {
        case JNothing   => default
        case JString(s) => s
        case other      => throw new IllegalArgumentException(s"$key must be string, got $other")
      }
  }
}