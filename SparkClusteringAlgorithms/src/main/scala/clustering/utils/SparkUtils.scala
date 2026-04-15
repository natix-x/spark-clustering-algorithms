package main.scala.clustering.utils

import org.apache.spark.rdd.RDD


object SparkUtils {

  def broadcastSafe[T](data: T)(implicit sc: org.apache.spark.SparkContext) =
    sc.broadcast(data)

  def repartitionIfNeeded[T](rdd: RDD[T], partitions: Int): RDD[T] =
    if (rdd.getNumPartitions != partitions) rdd.repartition(partitions)
    else rdd
}
