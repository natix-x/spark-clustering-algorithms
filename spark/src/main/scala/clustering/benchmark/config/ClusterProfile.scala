package clustering.benchmark.config

sealed trait ClusterProfile extends Serializable {
  def name: String

  def sparkMaster: Option[String]
}

case object LocalProfile extends ClusterProfile {
  val name: String = "local"
  val sparkMaster: Option[String] = Some("local[*]")
}

case object AresProfile extends ClusterProfile {
  val name: String = "ares"
  val sparkMaster: Option[String] = None
}

object ClusterProfile {

  def fromName(name: String): ClusterProfile = name.toLowerCase match {
    case "local" => LocalProfile
    case "ares"  => AresProfile
    case other   => throw new IllegalArgumentException(s"Unknown cluster profile: $other")
  }
}
