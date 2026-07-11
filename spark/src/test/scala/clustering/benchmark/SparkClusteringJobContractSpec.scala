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
  private def config(runId: String, algorithm: String): RunConfig = {
    val json =
      s"""{
         |  "runId": "$runId",
         |  "profile": "local",
         |  "dataset": { "type": "synthetic",
         |    "params": { "numPoints": 2000, "numPartitions": 2, "seed": 42 } },
         |  "algorithm": { "name": "$algorithm",
         |    "params": { "k": 5, "maxIter": 5, "distance": "euclidean", "seed": 42 } },
         |  "evaluation": { "metrics": ["silhouette", "nClusters", "clusterSizes", "noiseFraction"],
         |    "sampleSize": 500, "seed": 42 }
         |}""".stripMargin
    val tmp = Files.createTempFile(runId, ".json")
    tmp.toFile.deleteOnExit()
    Files.write(tmp, json.getBytes(StandardCharsets.UTF_8))
    RunConfig.fromFile(tmp.toString)
  }

  /** Assert a RunResult serialises to JSON that validates against the contract. */
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

  test("failed run (unknown algorithm) still emits a schema-conforming RunResult") {
    val result = new SparkClusteringJob().run(config("it-bad", "no-such-algo"), LocalProfile)
    assert(result.status == "failed", "expected the unknown-algorithm run to fail")
    assert(result.errorMessage.isDefined, "failed run must capture an error message")
    assertConformsToSchema(result)
  }
}
