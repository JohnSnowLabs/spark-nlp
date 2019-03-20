package com.johnsnowlabs.nlp

import org.apache.spark.sql.SparkSession

object SparkNLP {

  def start(): SparkSession = {
    SparkSession.builder()
      .appName("Spark NLP")
      .master("local[*]")
      .config("spark.driver.memory", "6G")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.jars.packages", "JohnSnowLabs:spark-nlp:2.0.0")
      .getOrCreate()
  }

}
