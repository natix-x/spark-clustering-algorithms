package clustering.algorithms.dbscan

import clustering.algorithms.dbscan.components.EpsilonGraphComponents
import clustering.distance.EuclideanDistance
import org.apache.spark.SparkConf
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.SparkSession
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

import scala.util.Random

/** The ε-graph phase runs as one distributed Spark job (no driver-local fallback — always worth
 *  its fixed cost per the 5.09.2026 decision to always pay for the distributed path). Checked
 *  against components computed from the definition (a plain BFS over the ε-graph), so "correct"
 *  cannot mean "wrong in a way the test shares".
 */
class EpsilonGraphComponentsSpec extends AnyFunSuite with BeforeAndAfterAll {

  private var spark: SparkSession = _

  override def beforeAll(): Unit = {
    spark = SparkSession.builder()
      .master("local[2]")
      .appName("epsilon-graph-spec")
      .config("spark.ui.enabled", "false")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
  }

  override def afterAll(): Unit = if (spark != null) spark.stop()

  /** Reference components straight from the definition: BFS over the ε-graph, labelled by
   *  ascending minimum member index (the same numbering `UnionFind.componentIds` produces). */
  private def componentsByDefinition(cores: Array[Vector], eps: Double): Array[Int] = {
    val labels = Array.fill(cores.length)(-1)
    var next = 0
    cores.indices.foreach { start =>
      if (labels(start) < 0) {
        val queue = scala.collection.mutable.Queue(start)
        labels(start) = next
        while (queue.nonEmpty) {
          val current = queue.dequeue()
          cores.indices.foreach { other =>
            if (labels(other) < 0 && EuclideanDistance.compute(cores(current), cores(other)) <= eps) {
              labels(other) = next
              queue.enqueue(other)
            }
          }
        }
        next += 1
      }
    }
    labels
  }

  /** Three well-separated blobs plus a few stragglers, so there is more than one component and
   *  the labelling has something to get wrong. */
  private def cores(count: Int, seed: Long): Array[Vector] = {
    val random  = new Random(seed)
    val centres = Array((0.0, 0.0), (20.0, 0.0), (0.0, 20.0))
    Array.tabulate(count) { i =>
      val (cx, cy) = centres(i % centres.length)
      Vectors.dense(cx + random.nextDouble(), cy + random.nextDouble())
    }
  }

  private val eps = 1.5

  test("distributed path matches components computed from the definition") {
    val points = cores(120, seed = 7L)
    assert(EpsilonGraphComponents.computeDistributed(points, eps, EuclideanDistance, spark.sparkContext).toSeq ==
      componentsByDefinition(points, eps).toSeq)
  }

  test("block size does not change the labels") {
    val points = cores(1200, seed = 5L)
    val reference = EpsilonGraphComponents.computeDistributed(points, eps, EuclideanDistance, spark.sparkContext, rowsPerBlock = 1).toSeq
    Seq(1, 7, 256, 1200, 5000).foreach { rowsPerBlock =>
      assert(
        EpsilonGraphComponents.computeDistributed(points, eps, EuclideanDistance, spark.sparkContext, rowsPerBlock).toSeq
          == reference,
        s"labels changed at rowsPerBlock=$rowsPerBlock")
    }
  }

  private val megabyte = 1024L * 1024

  /** The dense graph that used to OOM the distributed path (1.9 M cores, ~10⁴ neighbours per row,
   *  a naive collect of the raw edges) no longer needs a size-based refusal: [[computeDistributed]]'s
   *  per-partition local union-find ships only structural merges, bounded by vertex count rather
   *  than by how dense the component is. Correctness of that compaction — not the density itself
   *  — is what a near-clique exercises below. */
  test("a dense near-clique still labels correctly, via the compacted edges") {
    // All points in one tight blob: almost every pair is within eps, so a naive per-row edge
    // list would be O(m^2) — this is the compaction's actual job.
    val points = Array.tabulate(2000)(i => Vectors.dense(0.001 * (i % 7), 0.001 * (i / 7)): Vector)
    assert(EpsilonGraphComponents.computeDistributed(points, eps, EuclideanDistance, spark.sparkContext, rowsPerBlock = 500)
      .toSeq == componentsByDefinition(points, eps).toSeq)
  }

  test("a single core point is its own component") {
    val one = Array(Vectors.dense(1.0, 1.0): Vector)
    assert(EpsilonGraphComponents.computeDistributed(one, eps, EuclideanDistance, spark.sparkContext).toSeq == Seq(0))
  }

  test("compute matches components computed from the definition") {
    val points = cores(200, seed = 3L)
    assert(EpsilonGraphComponents.compute(points, eps, EuclideanDistance, spark.sparkContext).toSeq ==
      componentsByDefinition(points, eps).toSeq)
  }

  test("the default broadcast cap scales with configured executor memory") {
    def capFor(executorMemory: String): Long =
      EpsilonGraphComponents.getMaxBroadcastBytes(new SparkConf().set("spark.executor.memory", executorMemory))

    assert(capFor("4g") == (4L * 1024 * 1024 * 1024 * 0.3).toLong, s"expected 30% of 4g, got ${capFor("4g")}")
    assert(capFor("1000g") == (1000L * 1024 * 1024 * 1024 * 0.3).toLong,
      s"expected 30% of 1000g uncapped, got ${capFor("1000g")}")
  }
}
