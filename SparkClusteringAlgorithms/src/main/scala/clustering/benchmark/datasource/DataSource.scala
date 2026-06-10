package clustering.benchmark.datasource

import clustering.benchmark.config.DataSourceSpec
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.json4s._

/** Source of a `DataFrame` (single `features` Vector column) for one
 *  benchmark run.
 *
 *  Each impl owns its loading details (synthetic generator, Parquet reader,
 *  CSV reader, ...). Format-comparison experiments swap impls behind this
 *  trait; everything downstream (algorithm, evaluator, metrics) is unaware.
 *
 *  `metadata` is merged into the final `RunResult` so the per-run JSON
 *  captures whatever the source considers worth recording (file path, schema,
 *  partition count, ...).
 */
trait DataSource extends Serializable {
  def name: String
  def load(spark: SparkSession): DataFrame
  def metadata: Map[String, String]
}

object DataSource {

  trait Factory {
    def typeName: String
    def create(params: JObject): DataSource
  }

  // Built-in registry; populated lazily so impls can register themselves below.
  private var factories: Map[String, Factory] = Map.empty

  def register(factory: Factory): Unit = synchronized {
    factories = factories.updated(factory.typeName.toLowerCase, factory)
  }

  def create(spec: DataSourceSpec): DataSource =
    factories.get(spec.`type`.toLowerCase) match {
      case Some(f) => f.create(spec.params)
      case None =>
        throw new IllegalArgumentException(
          s"Unknown data source type: '${spec.`type`}'. " +
          s"Known: ${factories.keys.toSeq.sorted.mkString(", ")}"
        )
    }

  def known: Set[String] = factories.keySet

  // Built-ins
  register(SyntheticDataSource.Factory)
  register(ParquetDataSource.Factory)
  register(CsvDataSource.Factory)
}