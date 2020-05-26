package com.johnsnowlabs.nlp

import org.apache.spark.sql.SparkSession

object SparkNLP {

  val currentVersion = "2.5.1"

  def start(gpu:Boolean = false): SparkSession = {
    val build = SparkSession.builder()
      .appName("Spark NLP")
      .master("local[*]")
      .config("spark.driver.memory", "16G")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.kryoserializer.buffer.max", "1000M")
    
    if(gpu){
      build.config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp-gpu_2.11:2.5.1")
    }
    else
      build.config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.11:2.5.1")

    build.getOrCreate()
  }

  def version(): String = {
    currentVersion
  }

}
