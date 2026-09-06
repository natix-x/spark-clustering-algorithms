package clustering.benchmark.registry

import clustering.core.{EuclideanGeometry, Geometry, SphericalGeometry}

/** Maps the `geometry` param to a [[Geometry]] — mirrors [[DistanceRegistry]] so
 *  every config string is resolved the same way, with the same "unknown X" error. */
object GeometryRegistry {

  /** Used when a config omits `geometry`, keeping old configs' behaviour unchanged. */
  val Default: Geometry = EuclideanGeometry

  private val registry: NamedRegistry[Geometry] =
    NamedRegistry("geometry", Seq(
      EuclideanGeometry.name -> EuclideanGeometry,
      SphericalGeometry.name -> SphericalGeometry
    ))

  def knownNames: Seq[String] = registry.knownNames

  def get(name: String): Geometry = registry.get(name)
}
