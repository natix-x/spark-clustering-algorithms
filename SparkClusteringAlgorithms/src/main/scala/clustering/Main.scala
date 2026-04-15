package clustering

import clustering.core.Model
import clustering.data.Point
import clustering.distance.EuclideanDistance
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession


object Main {

  def main(args: Array[String]): Unit = {

    // 1. Spark session
    val spark = SparkSession.builder()
      .appName("ClusteringTest")
      .master("local[*]")
      .getOrCreate()

    val sc = spark.sparkContext

    // 2. Dummy dataset
    val data: RDD[Point] = sc.parallelize(Seq(
      Point(Vector(1.0, 2.0, -2.0)),
      Point(Vector(2.0, 1.0, 70.0)),
      Point(Vector(8.0, 9.0, 80.0)),
      Point(Vector(9.0, 8.0, 10.0))
    ))

    // 3. Dummy model (żeby sprawdzić pipeline)
    val model = new Model {
      override def predict(point: Point): Int = {
        if (point.values.sum > 10) 1 else 0
      }
    }

    // 4. Test model prediction
    println("=== MODEL TEST ===")
    data.collect().foreach { p =>
      println(s"${p.values} -> cluster ${model.predict(p)}")
    }

    // 5. Test distance metric
    println("\n=== DISTANCE TEST ===")
    val dist = new EuclideanDistance()

    val p1 = Point(Vector(1.0, 2.0))
    val p2 = Point(Vector(4.0, 6.0))

    println(s"Distance: ${dist.compute(p1, p2)}")

    // 6. Fake convergence test
    println("\n=== CONVERGENCE TEST ===")

    val oldC = Array(Point(Vector(1.0, 1.0)), Point(Vector(10.0, 10.0)))
    val newC = Array(Point(Vector(1.1, 1.1)), Point(Vector(9.9, 9.9)))

    val eps = 0.5

    val converged = oldC.zip(newC).forall {
      case (a, b) =>
        dist.compute(a, b) < eps
    }

    println(s"Converged: $converged")

    // 7. Stop Spark
    spark.stop()
  }
}
