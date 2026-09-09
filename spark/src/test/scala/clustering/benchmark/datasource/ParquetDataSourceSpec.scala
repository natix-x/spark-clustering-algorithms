package clustering.benchmark.datasource

import java.nio.file.Files

import clustering.core.Columns
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.col
import org.json4s.JsonDSL._
import org.json4s._
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

/** Pins [[ParquetDataSource]] to the layouts the Python preprocessing jobs actually write:
 *  Gaia/NYC emit `features : array[double]`, the three embedding sets emit `emb : array[float]`.
 *  Both must load through the same code path, only the column name differing.
 */
class ParquetDataSourceSpec extends AnyFunSuite with BeforeAndAfterAll {

  // A stable identifier (not a var) so `import spark.implicits._` compiles inside the tests.
  private lazy val spark: SparkSession = SparkSession.builder()
    .master("local[2]")
    .appName("parquet-datasource-spec")
    .config("spark.ui.enabled", "false")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()

  private var tmpDir: java.nio.file.Path = _

  override def beforeAll(): Unit = {
    spark.sparkContext.setLogLevel("WARN")
    tmpDir = Files.createTempDirectory("parquet-ds-spec")
  }

  override def afterAll(): Unit = {
    spark.stop()
    if (tmpDir != null) deleteRecursively(tmpDir.toFile)
  }

  private def deleteRecursively(f: java.io.File): Unit = {
    if (f.isDirectory) Option(f.listFiles()).toSeq.flatten.foreach(deleteRecursively)
    f.delete()
  }

  private def writeTabular(name: String, numRows: Int, dim: Int): String = {
    import spark.implicits._
    val path = tmpDir.resolve(name).toString
    (0 until numRows).map(i => Array.tabulate(dim)(j => (i + j).toDouble))
      .toDF("features").write.mode("overwrite").parquet(path)
    path
  }

  private def writeEmbeddings(name: String, numRows: Int, dim: Int): String = {
    import spark.implicits._
    val path = tmpDir.resolve(name).toString
    (0 until numRows).map(i => Array.tabulate(dim)(j => (i + j).toFloat))
      .toDF("emb").write.mode("overwrite").parquet(path)
    path
  }

  private def dims(df: DataFrame): Set[Int] =
    df.select(col(Columns.Features)).collect().map(_.getAs[Vector](0).size).toSet

  test("loads the 'features' array<double> column written by the Gaia/NYC jobs") {
    val path = writeTabular("gaia_like", numRows = 20, dim = 8)
    val df = new ParquetDataSource(path, "features").load(spark)

    assert(df.columns.toSeq == Seq(Columns.Features))
    assert(df.count() == 20)
    assert(dims(df) == Set(8))
  }

  test("loads 'emb' and widens array<float> to a double Vector") {
    val path = writeEmbeddings("cohere_like", numRows = 10, dim = 16)
    val df = new ParquetDataSource(path, "emb").load(spark)

    assert(df.count() == 10)
    assert(dims(df) == Set(16))
    // Row i is [i, i+1, ...]; the float -> double widening must be exact for these values.
    // Parquet read order is not guaranteed, so every vector is checked against its own head.
    val vecs = df.collect().map(_.getAs[Vector](0).toArray)
    assert(vecs.map(_.head).sorted.sameElements((0 until 10).map(_.toDouble)))
    vecs.foreach(v => assert(v.sameElements(Array.tabulate(16)(j => v.head + j))))
  }

  test("weight column is carried through as Columns.Weight") {
    import spark.implicits._
    val path = tmpDir.resolve("weighted").toString
    (1 to 4).map(i => (Array(i.toDouble, 0.0), i.toDouble))
      .toDF("features", "w").write.mode("overwrite").parquet(path)

    val df = new ParquetDataSource(path, "features", weightColumnName = Some("w")).load(spark)
    assert(df.columns.toSet == Set(Columns.Features, Columns.Weight))
    assert(df.select(Columns.Weight).as[Double].collect().sum == 10.0)
  }

  test("sampleFraction shrinks the row count reproducibly") {
    val path = writeTabular("sampled", numRows = 1000, dim = 3)

    def sampled = new ParquetDataSource(path, "features", sampleFraction = Some(0.1), sampleSeed = 7L)
      .load(spark).count()

    val n = sampled
    assert(n > 0 && n < 1000, s"expected a proper subset, got $n")
    // Same seed and fraction must give the same subset — the thesis measures reproducibility.
    assert(sampled == n)
  }

  test("numPartitions coalesces down and shuffles up") {
    val path = writeTabular("repart", numRows = 200, dim = 2)
    assert(new ParquetDataSource(path, "features", targetPartitionCount = Some(7)).load(spark).rdd.getNumPartitions == 7)

    val down = new ParquetDataSource(path, "features", targetPartitionCount = Some(1)).load(spark)
    assert(down.rdd.getNumPartitions == 1)
    assert(down.count() == 200)
  }

  test("a non-numeric feature column is a config error") {
    import spark.implicits._
    val path = tmpDir.resolve("strings").toString
    Seq("a", "b").toDF("features").write.mode("overwrite").parquet(path)

    val e = intercept[IllegalArgumentException] {
      new ParquetDataSource(path, "features").load(spark)
    }
    assert(e.getMessage.contains("unsupported feature-column type"))
  }

  test("factory parses the config params the harness emits") {
    val params: JObject =
      ("path" -> "/data/gaia") ~ ("featureColumnName" -> "features") ~ ("sampleFraction" -> 0.01) ~
        ("numPartitions" -> 64) ~ ("weightColumn" -> "w") ~ ("seed" -> 7)

    val ds = ParquetDataSource.Factory.create(params).asInstanceOf[ParquetDataSource]
    assert(ds.path == "/data/gaia")
    assert(ds.featureColumnName == "features")
    assert(ds.sampleFraction.contains(0.01))
    assert(ds.targetPartitionCount.contains(64))
    assert(ds.weightColumnName.contains("w"))
    assert(ds.sampleSeed == 7L)
    assert(ds.metadata("sampleFraction") == "0.01")
  }

  test("factory rejects a config without path or featureColumnName") {
    intercept[IllegalArgumentException](ParquetDataSource.Factory.create("featureColumnName" -> "emb"))
    intercept[IllegalArgumentException](ParquetDataSource.Factory.create("path" -> "/data/gaia"))
  }
}