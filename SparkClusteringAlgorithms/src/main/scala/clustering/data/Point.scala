package clustering.data


case class Point(values: Vector[Double]) extends Serializable {
  def dimension: Int = values.size
}