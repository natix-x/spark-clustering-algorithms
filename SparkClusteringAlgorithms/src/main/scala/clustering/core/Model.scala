package main.scala.clustering.core

import main.scala.clustering.data.Point

trait Model[T] {
    def predict(point: Point): Int
}
