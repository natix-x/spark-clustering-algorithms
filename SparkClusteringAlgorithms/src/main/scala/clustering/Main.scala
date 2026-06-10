package clustering

import clustering.algorithms.dbscan.GridDBSCAN
import clustering.benchmark.datasource.SyntheticDataSource
import clustering.distance.EuclideanDistance
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

/** Quick local smoke test of the DataFrame-based pipeline.
 *  The real benchmark entry point is `clustering.benchmark.BenchmarkRunner`.
 */
object Main {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("DBSCAN-Thesis-Pipeline")
      .master("local[*]")
      .getOrCreate()

    val sc = spark.sparkContext
    sc.setLogLevel("WARN")
    // GraphFrames connected-components checkpoints its iterations.
    sc.setCheckpointDir(s"${System.getProperty("java.io.tmpdir")}/spark-checkpoints-main")

    println("=== ROZPOCZYNAM TEST GridDBSCAN (DENSITY-BASED CLUSTERING) ===")

    val numPoints = 200000
    println(s"\n[1/3] Generowanie $numPoints punktów (HARD MODE - ZŁOŻONE DANE)...")

    val data = new SyntheticDataSource(numPoints = numPoints, numPartitions = 13, seed = 42L)
      .load(spark)
      .cache()
    data.count() // wymuszenie akcji

    println("\n[2/3] Trenowanie GridDBSCAN...")
    val eps    = 10.0
    val minPts = 50
    val trainer = new GridDBSCAN(eps = eps, minPts = minPts, distance = EuclideanDistance)

    val t0    = System.nanoTime()
    val model = trainer.fit(data)
    val t1    = System.nanoTime()
    println(f"-> GridDBSCAN zakończony w czasie: ${(t1 - t0) / 1e9d}%.3f s")

    println("\n[3/3] Podsumowanie rozkładu klastrów...")
    val distribution = model.labeledData(data)
      .groupBy(col("prediction"))
      .count()
      .orderBy(col("prediction"))
      .collect()

    println("\nRozkład klastrów GridDBSCAN:")
    distribution.foreach { r =>
      val clusterId = r.getInt(0)
      val count     = r.getLong(1)
      if (clusterId == -1) println(s"  NOISE (szum): $count punktów")
      else println(s"  Klaster $clusterId: $count punktów")
    }

    val numClusters = distribution.count(_.getInt(0) >= 0)
    println(s"\nZnaleziono $numClusters klastrów (+ szum)")

    spark.stop()
    println("\n=== TEST ZAKOŃCZONY SUKCESEM ===")
  }
}
