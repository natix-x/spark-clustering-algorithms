package clustering.utils

import org.apache.spark.rdd.RDD

/** Folds each partition's rows into a fixed-length array — one accumulator per partition,
 *  mutated in place — then merges the partials with `treeAggregate` (bounded driver memory,
 *  unlike collect-then-sum).
 *
 *  Merge order is task-completion order: exact for unit weights (integer sums), so
 *  unweighted runs are bit-reproducible at a fixed partition count; fractional weights can
 *  differ in the last bits.
 *
 *  Use only for a reduction into a long array behind an opaque UDF; everything else stays
 *  in Catalyst.
 */
object PartitionAggregator {

  /** @param length accumulator length; the same for every partition
   *  @param depth  `treeAggregate` levels */
  def aggregateDoubles[A](data: RDD[A], length: Int, depth: Int = 2)
                         (fold: (Array[Double], A) => Unit): Array[Double] =
    data.treeAggregate(new Array[Double](length))(
      seqOp = (acc, row) => { fold(acc, row); acc },
      // In-place merge into the left accumulator - allocates nothing.
      combOp = (a, b) => {
        var i = 0
        while (i < length) { a(i) += b(i); i += 1 }
        a
      },
      depth = depth
    )
}
