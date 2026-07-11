package clustering.benchmark.registry

import clustering.benchmark.config.DataSourceSpec
import clustering.benchmark.datasource.{DataSource, ParquetDataSource, SyntheticDataSource}

/** Maps a `DataSourceSpec.type` to the matching [[DataSource.Factory]]. Add a
 *  data source by listing its Factory here — mirrors [[AlgorithmRegistry]] and
 *  [[DistanceRegistry]] so all registration lives in one package. */
object DataSourceRegistry {

  private val registry: NamedRegistry[DataSource.Factory] =
    NamedRegistry("data source type", Seq(SyntheticDataSource.Factory, ParquetDataSource.Factory)
      .map(f => f.typeName -> f))

  def create(spec: DataSourceSpec): DataSource =
    registry.get(spec.`type`).create(spec.params)
}