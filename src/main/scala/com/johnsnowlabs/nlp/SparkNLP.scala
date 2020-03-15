package com.johnsnowlabs.nlp

import org.apache.spark.sql.SparkSession

object SparkNLP {

  val currentVersion = "2.4.4-rc1"

  def start(gpu:Boolean = false): SparkSession = {
    val build = SparkSession.builder()
      .appName("Spark NLP")
      .master("local[*]")
      .config("spark.driver.memory", "8G")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    
    if(gpu){
      build.config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp-gpu_2.11:2.4.4-rc1")
    }
    else
      build.config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.4-rc1")

    build.getOrCreate()
  }

  def version(): String = {
    currentVersion
  }

}
