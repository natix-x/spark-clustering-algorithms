package clustering.core

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}


trait Model extends Serializable {

  def predict(features: Vector): Int

  def labeledData(data: DataFrame): DataFrame = {
    val self       = this
    val predictUDF = udf((f: Vector) => self.predict(f))
    data.withColumn("prediction", predictUDF(col("features")))
  }
}