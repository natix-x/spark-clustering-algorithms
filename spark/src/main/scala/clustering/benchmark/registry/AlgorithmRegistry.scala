package clustering.benchmark.registry

import clustering.algorithms.dbscan.DBSCANpp
import clustering.algorithms.dbscan.components.CandidateSelectionStrategy
import clustering.algorithms.kmeans.hierarchical.BisectingKMeans
import clustering.algorithms.kmeans.{BreathingKMeans, KMeans}
import clustering.algorithms.kmedoids.distributed.DistributedFastPAM
import clustering.algorithms.kmedoids.hybrid.{CLARA, PAMAE}
import clustering.benchmark.config.{AlgorithmSpec, Params}
import clustering.core.{Clusterer, Geometry}
import clustering.distance.DistanceMetric
import org.json4s._

object AlgorithmRegistry {

  /** A built clusterer together with the distance metric it was configured with.
   *  Returned by [[create]] so evaluation reuses the exact same metric instead of
   *  re-parsing the config a second time. */
  final case class Built(clusterer: Clusterer, distance: DistanceMetric)

  object AlgorithmName {
    val KMeans = "kmeans"
    val BisectingKMeans = "bisectingkmeans"
    val DistFastPAM = "distfastpam"
    val CLARA = "clara"
    val PAMAE = "pamae"
    val DBSCANpp = "dbscanpp"
  }

  private trait AlgorithmFactory {
    def name: String
    /** Builds the clusterer and states the metric it actually runs with. For the
     *  k-means family the metric is fixed by `geometry` (see [[clustering.core.Geometry]]);
     *  every other algorithm reads the config's `distance` param. */
    def create(params: Params): Built
  }

  private val registry: NamedRegistry[AlgorithmFactory] =
    NamedRegistry("algorithm", Seq(
      KMeansFactory,
      BisectingKMeansFactory,
      DistributedFastPAMFactory,
      CLARAFactory,
      PAMAEFactory,
      DBSCANppFactory
    ).map(f => f.name -> f))

  def create(spec: AlgorithmSpec): Built =
    registry.get(spec.name).create(Params(spec.params))

  /** Parses the required `distance` param. No default — every run must state its
   *  distance explicitly so benchmark results are unambiguous. Not used by the
   *  k-means family, whose metric is fixed by `geometry` instead. */
  private def distanceFrom(p: Params): DistanceMetric = p.opt("distance") match {
    case Some(JString(name)) => DistanceRegistry.get(name)
    case None => throw new IllegalArgumentException(
      s"Missing required 'distance' parameter. Known: ${DistanceRegistry.knownNames.mkString(", ")}"
    )
    case Some(other) => throw new IllegalArgumentException(s"'distance' must be a string, got: $other")
  }

  private def assignWithinEps(assign: String): Boolean = assign.toLowerCase match {
    case "eps" => true
    case "closest" => false
    case other => throw new IllegalArgumentException(s"Unknown assign mode: '$other'. Known: closest, eps")
  }

  /** Parses the optional `geometry` param (the space a centroid algorithm optimises
   *  in). Absent = euclidean, so existing configs keep their meaning. */
  private def geometryFrom(p: Params): Geometry =
    GeometryRegistry.get(p.stringOpt("geometry", GeometryRegistry.Default.name))

  // --- factories -----------------------------------------------------------

  private object KMeansFactory extends AlgorithmFactory {
    val name: String = AlgorithmName.KMeans

    /** `refine: none | breathing` — a knob on slot 1, so both share this registry entry and
     *  therefore the same params. `breathing` swaps the plain Lloyd run for Fritzke's
     *  add/remove cycle around the identical loop; `m0` is its breath size.
     *
     *  Metric is fixed by `geometry`, not by a config `distance` field — the centroid
     *  update is an arithmetic mean, only geometry-consistent under that geometry's own
     *  metric (see [[clustering.core.Geometry]]). */
    def create(p: Params): Built = {
      val k  = p.int("k")
      val maxIter = p.intOpt("maxIter", 100)
      val eps = p.doubleOpt("eps", 1e-4)
      val seed = p.longOpt("seed", 42L)
      val geometry = geometryFrom(p)

      val clusterer = p.stringOpt("refine", "none").toLowerCase match {
        case "none" =>
          new KMeans(k, maxIter, eps, seed, geometry)
        case "breathing" =>
          new BreathingKMeans(k, p.intOpt("m0", 5), maxIter, eps, seed, geometry, p.intOpt("maxCycles", 100))
        case other =>
          throw new IllegalArgumentException(s"Unknown refine mode: '$other'. Known: breathing, none")
      }

      Built(clusterer, geometry.modelDistance)
    }
  }

  private object BisectingKMeansFactory extends AlgorithmFactory {
    val name: String = AlgorithmName.BisectingKMeans

    def create(p: Params): Built = {
      val geometry = geometryFrom(p)
      val clusterer = new BisectingKMeans(
        k = p.int("k"),
        maxIter  = p.intOpt("maxIter", 20),
        eps = p.doubleOpt("eps", 1e-4),
        seed = p.longOpt("seed", 42L),
        geometry = geometry,
        // Steinbach et al.'s ITER: competing 2-means runs per split, cheapest wins.
        trials = p.intOpt("trials", 1),
        // Which leaf gets split: 'cost' minimises error (scikit-learn's default), 'size' balances the tree.
        select = p.stringOpt("select", "cost")
      )
      Built(clusterer, geometry.modelDistance)
    }
  }

  private object DistributedFastPAMFactory extends AlgorithmFactory {
    val name: String = AlgorithmName.DistFastPAM

    def create(p: Params): Built = {
      val distance = distanceFrom(p)
      Built(new DistributedFastPAM(
        k = p.int("k"),
        maxIter = p.intOpt("maxIter", 100),
        distance = distance
      ), distance)
    }
  }

  private object CLARAFactory extends AlgorithmFactory {
    val name: String = AlgorithmName.CLARA

    def create(p: Params): Built = {
      val distance = distanceFrom(p)
      Built(new CLARA(
        k = p.int("k"),
        numSamples = p.intOpt("numSamples", 5),
        sampleSize = p.intOpt("sampleSize", 1000),
        maxIter = p.intOpt("maxIter", 100),
        distance = distance,
        // Which driver-local solver runs on each sample (Schubert & Rousseeuw 2021).
        inner = p.stringOpt("inner", "fastpam"),
        seed = p.longOpt("seed", 42L)
      ), distance)
    }
  }

  /** PAMAE (KDD 2017) — parallel seeding (= CLARA) + parallel refinement over entire data. */
  private object PAMAEFactory extends AlgorithmFactory {
    val name: String = AlgorithmName.PAMAE

    def create(p: Params): Built = {
      val distance = distanceFrom(p)
      Built(new PAMAE(
        k = p.int("k"),
        numSamples = p.intOpt("numSamples", 5),
        sampleSize = p.intOpt("sampleSize", 1000),
        maxIter = p.intOpt("maxIter", 100),
        refineIters = p.intOpt("refineIters", 1),
        poolSize = p.intOpt("poolSize", 2000),
        inner = p.stringOpt("inner", "fastpam"),
        distance = distance,
        seed = p.longOpt("seed", 42L)
      ), distance)
    }
  }

  private object DBSCANppFactory extends AlgorithmFactory {
    val name: String = AlgorithmName.DBSCANpp

    def create(p: Params): Built = {
      val distance = distanceFrom(p)
      Built(new DBSCANpp(
        eps = p.double("eps"),
        minPts = p.int("minPts"),
        // The universal accuracy-vs-cost knob; required, so no run hides which m it used.
        coreSampleFraction = p.double("coreSampleFraction"),
        samplingStrategy = CandidateSelectionStrategy.fromName(
          p.stringOpt("sampling", "uniform"), p.intOpt("poolFactor", 4)
        ),
        // assign: 'eps' = classic DBSCAN noise semantics, 'closest' = the paper's rule.
        requireWithinEps = assignWithinEps(p.stringOpt("assign", "eps")),
        distanceMetric = distance,
        seed = p.longOpt("seed", 42L)
      ), distance)
    }
  }
}
