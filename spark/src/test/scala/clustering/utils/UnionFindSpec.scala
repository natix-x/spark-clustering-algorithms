package clustering.utils

import org.scalatest.funsuite.AnyFunSuite

/** Union-find is what replaces the distributed connected-components pass, so its two
 *  guarantees are worth pinning: correct components, and MINIMAL roots (which is what
 *  makes density cluster ids reproducible instead of arbitrary). */
class UnionFindSpec extends AnyFunSuite {

  test("find returns the smallest index in the component regardless of union order") {
    val uf = new UnionFind(6)
    uf.union(4, 2)
    uf.union(5, 4)
    uf.union(3, 1)
    assert(uf.find(5) == 2)
    assert(uf.find(2) == 2)
    assert(uf.find(3) == 1)
    assert(uf.find(0) == 0)
  }

  test("componentIds are contiguous and ordered by ascending smallest member") {
    val uf = new UnionFind(7)
    uf.union(6, 3)      // component {3, 6}
    uf.union(5, 1)      // component {1, 5}
    uf.union(4, 1)      // component {1, 4, 5}
    val ids = uf.componentIds()

    assert(ids.toSeq == Seq(0, 1, 2, 3, 1, 1, 3))
    assert(ids.distinct.sorted.toSeq == (0 until ids.distinct.length))
  }

  test("a graph with no edges gives one component per element") {
    assert(new UnionFind(4).componentIds().toSeq == Seq(0, 1, 2, 3))
  }
}
