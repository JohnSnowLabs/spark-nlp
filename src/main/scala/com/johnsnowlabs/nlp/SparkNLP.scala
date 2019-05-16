package com.johnsnowlabs.nlp

import org.apache.spark.sql.SparkSession

object SparkNLP {

  def start(includeOcr: Boolean = false): SparkSession = {
    val build = SparkSession.builder()
      .appName("Spark NLP")
      .master("local[*]")
      .config("spark.driver.memory", "6G")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

    if (includeOcr) {
      build.config("spark.jars.packages", "JohnSnowLabs:spark-nlp:2.0.4,com.johnsnowlabs.nlp:spark-nlp-ocr_2.11:2.0.4")
    } else {
      build.config("spark.jars.packages", "JohnSnowLabs:spark-nlp:2.0.4")
    }

    build.getOrCreate()
  }

}
