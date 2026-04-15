package main.scala.clustering.core

trait Clusterer[T] {
    def fit(data: T): Model[T]
}
