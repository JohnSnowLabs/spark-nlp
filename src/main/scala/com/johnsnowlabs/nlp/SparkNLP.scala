package com.johnsnowlabs.nlp

import org.apache.spark.sql.SparkSession

object SparkNLP {

  val currentVersion = "2.3.4"

  def start(): SparkSession = {
    val build = SparkSession.builder()
      .appName("Spark NLP")
      .master("local[*]")
      .config("spark.driver.memory", "6G")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.jars.packages", "JohnSnowLabs:spark-nlp:2.3.4")

    build.getOrCreate()
  }

  def version(): String = {
    currentVersion
  }

}
