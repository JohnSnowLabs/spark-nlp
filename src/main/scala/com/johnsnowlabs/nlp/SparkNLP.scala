package com.johnsnowlabs.nlp

import org.apache.spark.sql.SparkSession

object SparkNLP {

  val currentVersion = "3.0.0"
  val MavenSpark30 = s"com.johnsnowlabs.nlp:spark-nlp_2.12:$currentVersion"
  val MavenGpuSpark30 = s"com.johnsnowlabs.nlp:spark-nlp-gpu_2.12:$currentVersion"
  val MavenSpark24 = s"com.johnsnowlabs.nlp:spark-nlp-spark24_2.11:$currentVersion"
  val MavenGpuSpark24 = s"com.johnsnowlabs.nlp:spark-nlp-gpu-spark24_2.11:$currentVersion"
  val MavenSpark23 = s"com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:$currentVersion"
  val MavenGpuSpark23 = s"com.johnsnowlabs.nlp:spark-nlp-gpu-spark23_2.11:$currentVersion"

  /**
    * Start SparkSession with Spark NLP
    * @param gpu start Spark NLP with GPU
    * @param spark23 start Spark NLP on Apache Spark 2.3.x
    * @param spark24 start Spark NLP on Apache Spark 2.4.x
    * @param memory set driver memory for SparkSession
    * @return SparkSession
    */
  def start(gpu: Boolean = false, spark23: Boolean = false, spark24: Boolean = false, memory: String = "16G"): SparkSession = {
    val build = SparkSession.builder()
      .appName("Spark NLP")
      .master("local[*]")
      .config("spark.driver.memory", memory)
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.kryoserializer.buffer.max", "2000M")
      .config("spark.driver.maxResultSize", "0")

    if(gpu & spark23){
      build.config("spark.jars.packages", MavenGpuSpark23)
    } else if(gpu & spark24){
      build.config("spark.jars.packages", MavenGpuSpark24)
    } else if(spark23){
      build.config("spark.jars.packages", MavenSpark23)
    } else if(spark24){
      build.config("spark.jars.packages", MavenSpark24)
    } else if(gpu){
      build.config("spark.jars.packages", MavenGpuSpark30)
    } else {
      build.config("spark.jars.packages", MavenSpark30)
    }

    build.getOrCreate()
  }

  def version(): String = {
    currentVersion
  }

}
