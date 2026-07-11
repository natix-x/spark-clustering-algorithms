package clustering.benchmark.registry

import clustering.algorithms.dbscan.{DBSCAN, GridDBSCAN}
import clustering.algorithms.kmeans.KMeans
import clustering.algorithms.kmedoids.{CLARA, DistributedFastPAM, FastPAM, PAM}
import clustering.benchmark.config.{AlgorithmSpec, Params}
import clustering.core.Clusterer
import clustering.distance.DistanceMetric
import org.json4s._

object AlgorithmRegistry {

  /** A built clusterer together with the distance metric it was configured with.
   *  Returned by [[create]] so evaluation reuses the exact same metric instead of
   *  re-parsing the config a second time. */
  final case class Built(clusterer: Clusterer, distance: DistanceMetric)

  private trait AlgorithmFactory {
    def name: String
    /** `distance` is resolved once by [[create]] (shared across all algorithms)
     *  and injected here, so no factory re-parses or forgets it. */
    def create(params: Params, distance: DistanceMetric): Clusterer
  }

  private val registry: NamedRegistry[AlgorithmFactory] =
    NamedRegistry("algorithm", Seq(
      KMeansFactory, PAMFactory, FastPAMFactory, DistributedFastPAMFactory, CLARAFactory,
      DBSCANFactory, GridDBSCANFactory
    ).map(f => f.name -> f))

  /** Resolves the distance once and builds the clusterer with it, returning both
   *  so the caller need not re-parse the config to evaluate with the same metric. */
  def create(spec: AlgorithmSpec): Built = {
    val distance = distanceFrom(spec.params)
    Built(registry.get(spec.name).create(Params(spec.params), distance), distance)
  }

  /** Parses the required `distance` param. No default — every run must state its
   *  distance explicitly so benchmark results are unambiguous. */
  private def distanceFrom(params: JObject): DistanceMetric =
    (params \ "distance") match {
      case JString(name) => DistanceRegistry.get(name)
      case JNothing      => throw new IllegalArgumentException(
        s"Missing required 'distance' parameter. Known: ${DistanceRegistry.knownNames.mkString(", ")}"
      )
      case other         => throw new IllegalArgumentException(s"distance must be a string, got: $other")
    }

  // --- factories -----------------------------------------------------------

  private object KMeansFactory extends AlgorithmFactory {
    val name = "kmeans"
    def create(p: Params, distance: DistanceMetric): Clusterer = new KMeans(
      k        = p.int("k"),
      maxIter  = p.intOpt("maxIter", 100),
      eps      = p.doubleOpt("eps", 1e-4),
      distance = distance,
      seed     = p.longOpt("seed", 42L)
    )
  }

  private object PAMFactory extends AlgorithmFactory {
    val name = "pam"
    def create(p: Params, distance: DistanceMetric): Clusterer = new PAM(
      k        = p.int("k"),
      maxIter  = p.intOpt("maxIter", 100),
      distance = distance
    )
  }

  private object FastPAMFactory extends AlgorithmFactory {
    val name = "fastpam"
    def create(p: Params, distance: DistanceMetric): Clusterer = new FastPAM(
      k        = p.int("k"),
      maxIter  = p.intOpt("maxIter", 100),
      distance = distance
    )
  }

  private object DistributedFastPAMFactory extends AlgorithmFactory {
    val name = "distfastpam"
    def create(p: Params, distance: DistanceMetric): Clusterer = new DistributedFastPAM(
      k        = p.int("k"),
      maxIter  = p.intOpt("maxIter", 100),
      distance = distance
    )
  }

  private object CLARAFactory extends AlgorithmFactory {
    val name = "clara"
    def create(p: Params, distance: DistanceMetric): Clusterer = new CLARA(
      k          = p.int("k"),
      numSamples = p.intOpt("numSamples", 5),
      sampleSize = p.intOpt("sampleSize", 1000),
      maxIter    = p.intOpt("maxIter", 100),
      distance   = distance
    )
  }

  private object DBSCANFactory extends AlgorithmFactory {
    val name = "dbscan"
    def create(p: Params, distance: DistanceMetric): Clusterer = new DBSCAN(
      eps      = p.double("eps"),
      minPts   = p.int("minPts"),
      distance = distance
    )
  }

  private object GridDBSCANFactory extends AlgorithmFactory {
    val name = "griddbscan"
    def create(p: Params, distance: DistanceMetric): Clusterer = new GridDBSCAN(
      eps      = p.double("eps"),
      minPts   = p.int("minPts"),
      distance = distance
    )
  }
}