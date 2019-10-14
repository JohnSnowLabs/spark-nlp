package com.johnsnowlabs.nlp

import org.apache.spark.sql.SparkSession

object SparkNLP {

  val currentVersion = "2.3.0-rc1"

  def start(includeOcr: Boolean = false, includeEval: Boolean): SparkSession = {
    val build = SparkSession.builder()
      .appName("Spark NLP")
      .master("local[*]")
      .config("spark.driver.memory", "6G")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

    val ocrPackage = "JohnSnowLabs:spark-nlp:2.3.0-rc1,com.johnsnowlabs.nlp:spark-nlp-ocr_2.11:2.3.0-rc1," +
      "javax.media.jai:com.springsource.javax.media.jai.core:1.1.3"

    val evalPackage = "JohnSnowLabs:spark-nlp:2.3.0-rc1,com.johnsnowlabs.nlp:spark-nlp-eval_2.11:2.3.0-rc1"

    val allPackages = "JohnSnowLabs:spark-nlp:2.3.0-rc1,com.johnsnowlabs.nlp:spark-nlp-ocr_2.11:2.3.0-rc1," +
      "javax.media.jai:com.springsource.javax.media.jai.core:1.1.3," +
      "JohnSnowLabs:spark-nlp:2.3.0-rc1,com.johnsnowlabs.nlp:spark-nlp-eval_2.11:2.3.0-rc1"

    if (includeOcr && !includeEval) {
      build
        .config("spark.jars.packages", ocrPackage)
        .config("spark.jars.repositories", "http://repo.spring.io/plugins-release")
    } else if (includeEval && !includeOcr) {
      build
        .config("spark.jars.packages", evalPackage)
        .config("spark.jars.repositories", "http://repo.spring.io/plugins-release")
    } else if (includeEval && includeOcr) {
      build
        .config("spark.jars.packages", allPackages)
        .config("spark.jars.repositories", "http://repo.spring.io/plugins-release")
    }
    else {
      build
        .config("spark.jars.packages", "JohnSnowLabs:spark-nlp:2.3.0-rc1")
    }

    build.getOrCreate()
  }

  def version(): Unit = {
    println(currentVersion)
  }

}
