package com.johnsnowlabs.nlp

import org.apache.spark.sql.SparkSession

object SparkNLP {

  val currentVersion = "2.5.2"
  def start(gpu:Boolean=false,isSpark23:Boolean=false): SparkSession = {
    val build = SparkSession.builder()
      .appName("Spark NLP")
      .master("local[*]")
      .config("spark.driver.memory", "16G")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.kryoserializer.buffer.max", "1000M")
      .config("spark.driver.maxResultSize", "4096G")
    
    if(gpu){
      if(isSpark23){
        build.config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp-gpu-spark23_2.11:2.5.2")
      }else{
        build.config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp-gpu_2.11:2.5.2")
      }

    }
    else {
      if(isSpark23){
        build.config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:2.5.2")
      }else{
        build.config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.11:2.5.2")
      }

    }

    build.getOrCreate()
  }

  def version(): String = {
    currentVersion
  }

}
