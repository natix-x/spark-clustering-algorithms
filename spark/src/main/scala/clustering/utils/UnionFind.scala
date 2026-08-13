package clustering.utils

/**
 *
 *  `union` attaches the larger root under the smaller (no union-by-rank), so `find(i)`
 *  returns the smallest index in i's component — makes derived cluster ids reproducible.
 */
final class UnionFind(size: Int) {

  private val parent: Array[Int] = Array.tabulate(size)(identity)

  /** Representative of `i`'s component: the smallest index in it. */
  def find(i: Int): Int = {
    var root = i
    while (parent(root) != root) root = parent(root)
    // Path compression, iterative.
    var node = i
    while (parent(node) != root) {
      val next = parent(node)
      parent(node) = root
      node = next
    }
    root
  }

  /** @return `true` if `a` and `b` were in different components (a real merge happened),
   *          `false` if they already shared a root (no-op) — callers that only need to ship the
   *          STRUCTURAL edges of a component (not every redundant one that touches it) filter on
   *          this. */
  def union(a: Int, b: Int): Boolean = {
    val ra = find(a)
    val rb = find(b)
    if (ra != rb) {
      if (ra < rb) parent(rb) = ra else parent(ra) = rb
      true
    } else false
  }

  /** Contiguous component ids from 0, numbered by ascending minimum member index — so
   *  element `0`'s component is always id 0. */
  def componentIds(): Array[Int] = {
    val roots = (0 until size).map(find)
    val order = roots.distinct.sorted.zipWithIndex.toMap
    roots.map(order).toArray
  }
}
