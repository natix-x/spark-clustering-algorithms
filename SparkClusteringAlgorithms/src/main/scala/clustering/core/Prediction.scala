package main.scala.clustering.core

import main.scala.clustering.data.Point

trait Predictable[Model] {
  def model: Model
  def predict(point: Point): Int =
    model.predict(point)
}
