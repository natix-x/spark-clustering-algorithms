package clustering.benchmark.registry

import clustering.distance.{CosineDistance, DistanceMetric, EuclideanDistance, ManhattanDistance}

object DistanceRegistry {

  private val metrics: Map[String, DistanceMetric] = Map(
    "euclidean" -> EuclideanDistance,
    "manhattan" -> ManhattanDistance,
    "cosine"    -> CosineDistance
  )

  val known: Set[String] = metrics.keySet

  def get(name: String): DistanceMetric =
    metrics.getOrElse(name.toLowerCase, throw new IllegalArgumentException(
      s"Unknown distance metric: '$name'. Known: ${known.toSeq.sorted.mkString(", ")}"
    ))
}
