package clustering.benchmark.datasource

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.json4s._

/** A loadable benchmark dataset. Implementations live in this package and
 *  register their [[DataSource.Factory]] with
 *  [[clustering.benchmark.registry.DataSourceRegistry]]. */
trait DataSource extends Serializable {
  def name: String
  def load(spark: SparkSession): DataFrame
  def metadata: Map[String, String]
}

object DataSource {

  /** Builds a [[DataSource]] from its config params. Add a source by defining
   *  a Factory and listing it in
   *  [[clustering.benchmark.registry.DataSourceRegistry]]. */
  trait Factory {
    def typeName: String
    def create(params: JObject): DataSource
  }
}