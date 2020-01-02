package com.johnsnowlabs.nlp

import org.apache.spark.sql.SparkSession

object SparkNLP {

  val currentVersion = "2.3.6"

  def start(): SparkSession = {
    val build = SparkSession.builder()
      .appName("Spark NLP")
      .master("local[*]")
      .config("spark.driver.memory", "6G")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.11:2.3.6")

    build.getOrCreate()
  }

  def version(): String = {
    currentVersion
  }

}
