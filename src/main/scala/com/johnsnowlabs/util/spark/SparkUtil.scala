package com.johnsnowlabs.util.spark

import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf

object SparkUtil {

  //Helper UDF function to flatten arrays for Spark < 2.4.0
  def flattenArrays: UserDefinedFunction = udf { (arrayColumn: Seq[Seq[String]]) =>
    arrayColumn.flatten.distinct
  }

}
