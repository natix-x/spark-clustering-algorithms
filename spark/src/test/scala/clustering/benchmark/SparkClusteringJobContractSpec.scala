package clustering.benchmark

import clustering.benchmark.config.{LocalProfile, RunConfig}
import clustering.benchmark.metrics.RunResult
import com.fasterxml.jackson.databind.ObjectMapper
import com.networknt.schema.{JsonSchemaFactory, SpecVersion}
import org.scalatest.funsuite.AnyFunSuite

import java.nio.charset.StandardCharsets
import java.nio.file.Files
import scala.jdk.CollectionConverters._

/** End-to-end contract test: actually run [[SparkClusteringJob]] on a tiny
 *  synthetic dataset and assert the REAL emitted RunResult validates against
 *  `contract/run_result.schema.json` (vendored as a test resource).
 *
 *  This exercises the real fill logic (listener snapshot, timings, eval metrics,
 *  serialisation) — the thing that runs when you launch the benchmark. It also
 *  guards Scala/schema drift: the schema is `additionalProperties: false`, so a
 *  new RunResult field not declared in the schema fails here.
 */
class SparkClusteringJobContractSpec extends AnyFunSuite {

  /** Write a per-run config to a temp file and load it exactly like production
   *  (`RunConfig.fromFile`), so the test drives the real file-reading path. */
  private def config(
    runId:           String,
    algorithm:       String,
    distance:        String = "euclidean",
    algorithmParams: String = "",
    metrics:         Seq[String] = Seq("silhouette", "nClusters", "clusterSizes", "noiseFraction")
  ): RunConfig = {
    val metricsJson = metrics.map(m => "\"" + m + "\"").mkString(", ")
    val json =
      s"""{
         |  "runId": "$runId",
         |  "profile": "local",
         |  "dataset": { "type": "synthetic",
         |    "params": { "numPoints": 2000, "numPartitions": 2, "seed": 42 } },
         |  "algorithm": { "name": "$algorithm",
         |    "params": { "k": 5, "maxIter": 5, "distance": "$distance", "seed": 42$algorithmParams } },
         |  "evaluation": { "metrics": [$metricsJson],
         |    "sampleSize": 500, "seed": 42 }
         |}""".stripMargin
    val tmp = Files.createTempFile(runId, ".json")
    tmp.toFile.deleteOnExit()
    Files.write(tmp, json.getBytes(StandardCharsets.UTF_8))
    RunConfig.fromFile(tmp.toString)
  }

  private def assertConformsToSchema(result: RunResult): Unit = {
    val schemaStream = getClass.getResourceAsStream("/run_result.schema.json")
    assert(schemaStream != null, "run_result.schema.json missing from test resources")
    val schema = JsonSchemaFactory
      .getInstance(SpecVersion.VersionFlag.V202012)
      .getSchema(schemaStream)

    val json       = RunResult.toJsonString(result)
    val violations = schema.validate(new ObjectMapper().readTree(json)).asScala.toSet
    assert(violations.isEmpty,
      s"RunResult violates the contract schema:\n${violations.mkString("\n")}\n\njson:\n$json")
  }

  test("real kmeans run emits a schema-conforming RunResult") {
    val result = new SparkClusteringJob().run(config("it-kmeans", "kmeans"), LocalProfile)
    assert(result.status == "ok", s"run failed: ${result.errorMessage.getOrElse("")}")
    assertConformsToSchema(result)
  }

  /** The `refine: breathing` knob on slot 1 — same registry entry, same params, plus `m0`. */
  test("real breathing kmeans run emits a schema-conforming RunResult") {
    val result = new SparkClusteringJob().run(
      config("it-kmeans-breathing", "kmeans",
        algorithmParams = """, "refine": "breathing", "m0": 2, "maxCycles": 6"""),
      LocalProfile)
    assert(result.status == "ok", s"run failed: ${result.errorMessage.getOrElse("")}")
    assert(result.nClusters.contains(5), s"breathing must end with k clusters, got ${result.nClusters}")
    assertConformsToSchema(result)
  }

  test("real bisectingkmeans run emits a schema-conforming RunResult with k clusters") {
    val result = new SparkClusteringJob().run(config("it-bisecting", "bisectingkmeans"), LocalProfile)
    assert(result.status == "ok", s"run failed: ${result.errorMessage.getOrElse("")}")
    assert(result.nClusters.contains(5), s"expected 5 leaves, got ${result.nClusters}")
    assertConformsToSchema(result)
  }

  test("real fasterpam run emits a schema-conforming RunResult") {
    val result = new SparkClusteringJob().run(config("it-fasterpam", "fasterpam"), LocalProfile)
    assert(result.status == "ok", s"run failed: ${result.errorMessage.getOrElse("")}")
    assertConformsToSchema(result)
  }

  /** The `geometry: spherical` knob on slot 1 — spherical k-means goes through the same
   *  runner, so the only thing to verify end-to-end is that the config resolves and the
   *  run produces a conforming result with the requested number of clusters. The config's
   *  `distance` field is irrelevant here (kept at its default "euclidean") — geometry alone
   *  fixes the metric for the k-means family, so there is nothing for `distance` to override
   *  or disagree with. */
  test("spherical kmeans (cosine) run emits a schema-conforming RunResult") {
    val result = new SparkClusteringJob().run(
      config("it-kmeans-spherical", "kmeans",
        algorithmParams = """, "geometry": "spherical""""),
      LocalProfile)
    assert(result.status == "ok", s"run failed: ${result.errorMessage.getOrElse("")}")
    assertConformsToSchema(result)
  }

  test("real pamae run emits a schema-conforming RunResult") {
    val result = new SparkClusteringJob().run(
      config("it-pamae", "pamae",
        algorithmParams = """, "numSamples": 2, "sampleSize": 200, "poolSize": 200, "inner": "fasterpam""""),
      LocalProfile)
    assert(result.status == "ok", s"run failed: ${result.errorMessage.getOrElse("")}")
    assertConformsToSchema(result)
  }

  /** The engine-portable density entry: no distributed connected components, so this is the
   *  DBSCAN that can appear in a cross-engine table. */
  test("real dbscanpp run emits a schema-conforming RunResult") {
    val result = new SparkClusteringJob().run(
      config("it-dbscanpp", "dbscanpp",
        algorithmParams = """, "eps": 5.0, "minPts": 5, "coreSampleFraction": 0.3"""),
      LocalProfile)
    assert(result.status == "ok", s"run failed: ${result.errorMessage.getOrElse("")}")
    assert(result.noiseFraction.isDefined, "a density run must report a noise fraction")
    assertConformsToSchema(result)
  }

  /** The centroid-based internal indices. They are opt-in (absent from the default metric
   *  list), so this is the run that proves the config string reaches the runner, both numbers
   *  are produced, and the schema declares them — the schema is `additionalProperties: false`,
   *  so a field added to RunResult but not to the contract fails right here. */
  test("kmeans run with the internal indices requested emits a schema-conforming RunResult") {
    val result = new SparkClusteringJob().run(
      config("it-kmeans-indices", "kmeans",
        metrics = Seq("nClusters", "daviesBouldin", "calinskiHarabasz")),
      LocalProfile)
    assert(result.status == "ok", s"run failed: ${result.errorMessage.getOrElse("")}")
    assert(result.daviesBouldin.exists(_ > 0.0), s"expected a positive DB index, got ${result.daviesBouldin}")
    assert(result.calinskiHarabasz.exists(_ > 0.0), s"expected a positive CH index, got ${result.calinskiHarabasz}")
    assert(result.silhouette.isEmpty, "silhouette was not requested and must be omitted")
    assert(result.silhouetteScoredPoints.isEmpty && result.silhouetteSampleClusters.isEmpty &&
      result.silhouetteUnscoredPoints.isEmpty,
      "the silhouette's sample counters must be absent exactly when the silhouette is")
    assertConformsToSchema(result)
  }

  /** The silhouette's sample counters ship WITH the score, never separately.
   *
   *  A silhouette without them cannot be read: the same number comes out whether the sample kept
   *  every cluster or lost half of them, and the achieved sample size is Binomial around
   *  `sampleSize` rather than equal to it. This also pins the eval phase to a single labelling
   *  scan — the label-stats pass supplies both the draw's denominator and the per-cluster masses,
   *  which is why `silhouetteSampleClusters` can be compared against `nClusters` at all. */
  test("a silhouette run reports what it was computed on") {
    val result = new SparkClusteringJob().run(config("it-kmeans-silhouette", "kmeans"), LocalProfile)
    assert(result.status == "ok", s"run failed: ${result.errorMessage.getOrElse("")}")
    assert(result.silhouette.isDefined)
    assert(result.silhouetteScoredPoints.exists(_ > 0), "the achieved sample size must be reported")
    assert(result.silhouetteUnscoredPoints.contains(0), "clean synthetic data must refuse no points")
    assert(result.silhouetteSampleClusters == result.nClusters,
      "this dataset fits the budget, so the sample must hold every cluster: " +
        s"${result.silhouetteSampleClusters} of ${result.nClusters}")
    assertConformsToSchema(result)
  }

  /** `nClusters` rides along with a silhouette-only config, and it is the one metric that does.
   *
   *  Not a convenience: `silhouetteSampleClusters` is a bare number until it is read against the
   *  full-data cluster count, and a shortfall between them is exactly the case where `b` was
   *  minimised over the surviving clusters and the score reads HIGH. The label-stats scan that
   *  answers it already runs for the draw's denominator, so gating it on an explicit
   *  `"nClusters"` cost the counter its meaning and saved nothing. `clusterSizes` and
   *  `noiseFraction` come off that same scan and stay opt-in — they are not needed to READ
   *  another number. */
  test("a silhouette-only config still reports nClusters, and nothing else off that scan") {
    val result = new SparkClusteringJob().run(
      config("it-kmeans-silhouette-only", "kmeans", metrics = Seq("silhouette")),
      LocalProfile)
    assert(result.status == "ok", s"run failed: ${result.errorMessage.getOrElse("")}")
    assert(result.silhouette.isDefined)
    assert(result.nClusters.contains(5),
      s"silhouetteSampleClusters is unreadable without nClusters, got ${result.nClusters}")
    assert(result.silhouetteSampleClusters == result.nClusters,
      s"${result.silhouetteSampleClusters} of ${result.nClusters}")
    assert(result.clusterSizes.isEmpty && result.noiseFraction.isEmpty,
      "the rest of the label-stats scan stays opt-in")
    assertConformsToSchema(result)
  }

  test("failed run (unknown algorithm) still emits a schema-conforming RunResult") {
    val result = new SparkClusteringJob().run(config("it-bad", "no-such-algo"), LocalProfile)
    assert(result.status == "failed", "expected the unknown-algorithm run to fail")
    assert(result.errorMessage.isDefined, "failed run must capture an error message")
    assertConformsToSchema(result)
  }
}
