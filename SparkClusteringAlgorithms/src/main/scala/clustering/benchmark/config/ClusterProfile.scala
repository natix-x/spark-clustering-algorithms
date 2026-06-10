package clustering.benchmark.config

/** Execution environment for a benchmark run.
 *
 *  Encapsulates everything that differs between running on a laptop and on a
 *  shared HPC cluster: result paths, Spark master default, and (later) sbatch
 *  defaults. Keeping this behind a sealed trait means the same JAR can run
 *  unchanged on either side — the matrix expander just picks a profile.
 */
sealed trait ClusterProfile extends Serializable {
  def name: String

  def outputDir: String

  /** Default Spark master, or None when spark-submit will set it (cluster mode). */
  def sparkMaster: Option[String]
}

final case class LocalProfile(
  outputDir: String = sys.props.getOrElse("user.dir", ".") + "/benchmark-results"
) extends ClusterProfile {
  val name: String = "local"
  val sparkMaster: Option[String] = Some("local[*]")
}

/** Cyfronet Ares profile.
 *
 *  Paths follow PLGrid conventions: `$SCRATCH` for hot per-run artifacts,
 *  the merger script (Phase 3) promotes them to `$PLG_GROUPS_STORAGE`.
 */
final case class AresProfile(
  outputDir: String = sys.env.get("SCRATCH") + "/clustering-runs"
) extends ClusterProfile {
  val name: String = "ares"
  val sparkMaster: Option[String] = None
}

object ClusterProfile {

  def fromName(name: String): ClusterProfile = name.toLowerCase match {
    case "local" => LocalProfile()
    case "ares"  => AresProfile()
    case other   => throw new IllegalArgumentException(s"Unknown cluster profile: $other")
  }

  /** Detect profile from environment. Honors `CLUSTERING_PROFILE` first, then
   *  falls back to SLURM detection. Useful so sbatch templates don't need to
   *  pass `--profile` explicitly.
   */
  def fromEnv(): ClusterProfile =
    sys.env.get("CLUSTERING_PROFILE").map(fromName).getOrElse {
      if (sys.env.contains("SLURM_JOB_ID")) AresProfile() else LocalProfile()
    }
}