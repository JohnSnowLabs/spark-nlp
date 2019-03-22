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

  def startWithOcr(): SparkSession = {
    SparkSession.builder()
      .appName("Spark NLP")
      .master("local[*]")
      .config("spark.driver.memory", "6G")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.jars.packages", "JohnSnowLabs:spark-nlp:2.0.0,com.johnsnowlabs.nlp:spark-nlp-ocr_2.11:2.0.0")
      .getOrCreate()
  }

}
