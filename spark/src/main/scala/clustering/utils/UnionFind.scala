package clustering.utils


object UnionFind {

  def labelComponents[T](nodes: Array[T], edges: Array[(T, T)]): Map[T, Int] = {
    val parent = scala.collection.mutable.Map(nodes.map(n => n -> n): _*)

    def find(start: T): T = {
      var x = start
      while (parent(x) != x) x = parent(x)
      var y = start
      while (y != x) {
        val next = parent(y)
        parent(y) = x
        y = next
      }
      x
    }

    def union(a: T, b: T): Unit = {
      val ra = find(a)
      val rb = find(b)
      if (ra != rb) parent(ra) = rb
    }

    edges.foreach { case (a, b) => union(a, b) }

    val roots     = nodes.map(find).distinct
    val rootLabel = roots.zipWithIndex.toMap
    nodes.map(n => n -> rootLabel(find(n))).toMap
  }
}
