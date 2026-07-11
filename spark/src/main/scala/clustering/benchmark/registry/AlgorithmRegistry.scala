package clustering.benchmark.registry

import clustering.algorithms.dbscan.{DBSCAN, GridDBSCAN}
import clustering.algorithms.kmeans.KMeans
import clustering.algorithms.kmedoids.{CLARA, DistributedFastPAM, FastPAM, PAM}
import clustering.benchmark.config.AlgorithmSpec
import clustering.core.Clusterer
import clustering.distance.{DistanceMetric, EuclideanDistance}
import org.json4s._

object AlgorithmRegistry {

  private implicit val formats: Formats = DefaultFormats

  trait AlgorithmFactory {
    def name: String
    def create(params: JObject): Clusterer
  }

  private val factories: Map[String, AlgorithmFactory] = Seq(
    KMeansFactory, PAMFactory, FastPAMFactory, DistributedFastPAMFactory, CLARAFactory,
    DBSCANFactory, GridDBSCANFactory
  ).map(f => f.name -> f).toMap

  val known: Set[String] = factories.keySet

  def create(spec: AlgorithmSpec): Clusterer =
    factories.get(spec.name.toLowerCase) match {
      case Some(f) => f.create(spec.params)
      case None =>
        throw new IllegalArgumentException(
          s"Unknown algorithm: '${spec.name}'. Known: ${factories.keys.toSeq.sorted.mkString(", ")}"
        )
    }

  // --- helpers shared across factories -------------------------------------

  private def distanceFrom(params: JObject, default: DistanceMetric = EuclideanDistance): DistanceMetric =
    (params \ "distance") match {
      case JString(name) => DistanceRegistry.get(name)
      case JNothing      => default
      case other         => throw new IllegalArgumentException(s"distance must be a string, got: $other")
    }

  private def intParam(params: JObject, key: String): Int =
    (params \ key) match {
      case JNothing => throw new IllegalArgumentException(s"Missing required Int parameter: '$key'")
      case v        => v.extract[Int]
    }

  private def intParamOpt(params: JObject, key: String, default: Int): Int =
    (params \ key) match {
      case JNothing => default
      case v        => v.extract[Int]
    }

  private def doubleParam(params: JObject, key: String): Double =
    (params \ key) match {
      case JNothing => throw new IllegalArgumentException(s"Missing required Double parameter: '$key'")
      case v        => v.extract[Double]
    }

  private def doubleParamOpt(params: JObject, key: String, default: Double): Double =
    (params \ key) match {
      case JNothing => default
      case v        => v.extract[Double]
    }

  private def longParamOpt(params: JObject, key: String, default: Long): Long =
    (params \ key) match {
      case JNothing => default
      case v        => v.extract[Long]
    }

  // --- factories -----------------------------------------------------------

  private object KMeansFactory extends AlgorithmFactory {
    val name = "kmeans"
    def create(params: JObject): Clusterer = new KMeans(
      k        = intParam(params, "k"),
      maxIter  = intParamOpt(params, "maxIter", 100),
      eps      = doubleParamOpt(params, "eps", 1e-4),
      distance = distanceFrom(params),
      seed     = longParamOpt(params, "seed", 42L)
    )
  }

  private object PAMFactory extends AlgorithmFactory {
    val name = "pam"
    def create(params: JObject): Clusterer = new PAM(
      k        = intParam(params, "k"),
      maxIter  = intParamOpt(params, "maxIter", 100),
      distance = distanceFrom(params)
    )
  }

  private object FastPAMFactory extends AlgorithmFactory {
    val name = "fastpam"
    def create(params: JObject): Clusterer = new FastPAM(
      k        = intParam(params, "k"),
      maxIter  = intParamOpt(params, "maxIter", 100),
      distance = distanceFrom(params)
    )
  }

  private object DistributedFastPAMFactory extends AlgorithmFactory {
    val name = "distfastpam"
    def create(params: JObject): Clusterer = new DistributedFastPAM(
      k        = intParam(params, "k"),
      maxIter  = intParamOpt(params, "maxIter", 100),
      distance = distanceFrom(params)
    )
  }

  private object CLARAFactory extends AlgorithmFactory {
    val name = "clara"
    def create(params: JObject): Clusterer = new CLARA(
      k          = intParam(params, "k"),
      numSamples = intParamOpt(params, "numSamples", 5),
      sampleSize = intParamOpt(params, "sampleSize", 1000),
      maxIter    = intParamOpt(params, "maxIter", 100),
      distance   = distanceFrom(params)
    )
  }

  private object DBSCANFactory extends AlgorithmFactory {
    val name = "dbscan"
    def create(params: JObject): Clusterer = new DBSCAN(
      eps      = doubleParam(params, "eps"),
      minPts   = intParam(params, "minPts"),
      distance = distanceFrom(params)
    )
  }

  private object GridDBSCANFactory extends AlgorithmFactory {
    val name = "griddbscan"
    def create(params: JObject): Clusterer = new GridDBSCAN(
      eps      = doubleParam(params, "eps"),
      minPts   = intParam(params, "minPts"),
      distance = distanceFrom(params)
    )
  }
}