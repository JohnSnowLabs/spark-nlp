package com.jsl.nlp

import org.apache.spark.sql.SparkSession

object SparkAccessor {

  val spark: SparkSession = SparkSession
    .builder()
    .appName("test")
    .master("local[*]")
    .config("spark.driver.memory","4G")
    .config("spark.kryoserializer.buffer.max","200M")
    .getOrCreate()
}