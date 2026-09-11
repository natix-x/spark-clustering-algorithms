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

  /** Same fold, but merged in PARTITION-INDEX order, so the result is bit-identical between two
   *  runs of the same configuration.
   *
   *  `treeAggregate` merges in task-COMPLETION order, which for floating-point sums means two
   *  identical runs can differ in the last bits. That is acceptable where a run's result is a
   *  choice among discrete options (a medoid index, a core point) and it is NOT acceptable for
   *  k-means, whose centroids are the sums themselves: `BisectingLeafSelectionSpec` compares two
   *  fits of the same configuration for equality, and more importantly the thesis reports
   *  "Spark's partitioning is deterministic, so this entry is reproducible there" as a RESULT
   *  against Flink. Buying that back costs one ordered merge.
   *
   *  Price: the partial arrays are COLLECTED, so the driver holds `mergeGroups × length` doubles.
   *  Negligible for Lloyd (k·(d+1)) but not at every dataset — at d=1024 and thousands of source
   *  partitions, one array PER PARTITION would be hundreds of MB/iteration on the driver;
   *  `mergeGroups` bounds that regardless of partition count. Do NOT use this for the long
   *  accumulators (`dbscanpp`'s n·m, DistributedFastPAM's n·k) — there the tree merge keeps the
   *  driver alive.
   */
  private val MaxMergeGroups = 256

  def aggregateDoublesOrdered[A](data: RDD[A], length: Int)
                                (fold: (Array[Double], A) => Unit): Array[Double] = {
    val partials = data.mapPartitionsWithIndex { (index, rows) =>
      val acc = new Array[Double](length)
      while (rows.hasNext) { fold(acc, rows.next()) }
      Iterator.single(index -> acc)
    }

    // Fan in BEFORE the collect, narrow (no shuffle, one job, every core stays busy): `coalesce`
    // groups partitions by a static function of (numPartitions, mergeGroups), never by timing, so
    // it stays as reproducible as a flat fold — just capped at `mergeGroups × length` doubles on
    // the driver. No-op (bit-identical) below MaxMergeGroups; only wide-partition runs get a
    // different, still-deterministic merge order.
    val mergeGroups = math.min(partials.getNumPartitions, MaxMergeGroups)
    val merged = partials.coalesce(mergeGroups, shuffle = false)
      .mapPartitionsWithIndex { (groupIndex, tuples) =>
        val acc = new Array[Double](length)
        tuples.toArray.sortBy(_._1).foreach { case (_, part) =>
          var i = 0
          while (i < length) { acc(i) += part(i); i += 1 }
        }
        Iterator.single(groupIndex -> acc)
      }.collect()

    val total = new Array[Double](length)
    merged.sortBy(_._1).foreach { case (_, part) =>
      var i = 0
      while (i < length) { total(i) += part(i); i += 1 }
    }
    total
  }
}
