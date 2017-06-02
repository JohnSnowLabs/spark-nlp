package com.jsl.nlp

import org.apache.spark.sql.SparkSession

object SparkAccessor {

  val spark: SparkSession = SparkSession
    .builder()
    .appName("test")
    .master("local[4]")
    .config("spark.driver.memory","512M")
    .getOrCreate()

}