package main.scala.clustering.distance

trait DistanceMetric extends Serializable {
    def compute(a: Point, b: Point): Double
}
