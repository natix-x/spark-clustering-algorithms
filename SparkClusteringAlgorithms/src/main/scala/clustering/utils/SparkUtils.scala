package clustering.utils

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import scala.reflect.ClassTag


object SparkUtils {

  def broadcastSafe[T: ClassTag](data: T)(implicit sc: SparkContext) =
    sc.broadcast(data)

  def repartitionIfNeeded[T: ClassTag](rdd: RDD[T], partitions: Int): RDD[T] =
    if (rdd.getNumPartitions != partitions) rdd.repartition(partitions)
    else rdd
}
