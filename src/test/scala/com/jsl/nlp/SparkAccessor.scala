package com.jsl.nlp

import org.apache.spark.sql.SparkSession

object SparkAccessor {

  val spark: SparkSession = SparkSession
    .builder()
    .appName("test")
    .master("local[8]")
    .config("spark.driver.memory","4G")
    .getOrCreate()

}