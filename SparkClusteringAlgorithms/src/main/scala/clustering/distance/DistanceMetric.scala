package clustering.distance

import clustering.data.Point


trait DistanceMetric extends Serializable {
    def compute(a: Point, b: Point): Double
}
