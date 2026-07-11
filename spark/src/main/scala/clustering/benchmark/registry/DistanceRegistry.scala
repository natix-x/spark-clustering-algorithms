package clustering.benchmark.registry

import clustering.distance.{CosineDistance, DistanceMetric, EuclideanDistance, ManhattanDistance}

object DistanceRegistry {

  private val registry: NamedRegistry[DistanceMetric] =
    NamedRegistry("distance metric", Seq(
      "euclidean" -> EuclideanDistance,
      "manhattan" -> ManhattanDistance,
      "cosine"    -> CosineDistance
    ))

  /** Sorted list of registered metric names, for error messages. */
  def knownNames: Seq[String] = registry.knownNames

  def get(name: String): DistanceMetric = registry.get(name)
}