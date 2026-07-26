package clustering.distance

import org.apache.spark.ml.linalg.{Vector, Vectors}

object CosineDistance extends DistanceMetric {

  override def compute(a: Vector, b: Vector): Double = {
    val normA = Vectors.norm(a, 2.0)
    val normB = Vectors.norm(b, 2.0)

    if (normA == 0.0 || normB == 0.0) {
      1.0
    } else {
      1.0 - (a.dot(b) / (normA * normB))
    }
  }
}
