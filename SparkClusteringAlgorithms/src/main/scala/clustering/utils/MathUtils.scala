package main.scala.clustering.utils

object MathUtils {

  def argmin[T](seq: Seq[T])(f: T => Double): Int = {
    seq.zipWithIndex.minBy { case (v, _) => f(v) }._2
  }
}
