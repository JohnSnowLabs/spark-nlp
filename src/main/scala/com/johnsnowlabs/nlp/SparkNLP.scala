package com.johnsnowlabs.nlp

import org.apache.spark.sql.SparkSession

object SparkNLP {

  val currentVersion = "3.0.0-rc1"
  val MavenSpark24 = s"com.johnsnowlabs.nlp:spark-nlp_2.11:$currentVersion"
  val MavenGpuSpark24 = s"com.johnsnowlabs.nlp:spark-nlp-gpu_2.11:$currentVersion"
  val MavenSpark23 = s"com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:$currentVersion"
  val MavenGpuSpark23 = s"com.johnsnowlabs.nlp:spark-nlp-gpu-spark23_2.11:$currentVersion"

  def start(gpu:Boolean = false, spark23:Boolean = false): SparkSession = {
    val build = SparkSession.builder()
      .appName("Spark NLP")
      .master("local[*]")
      .config("spark.driver.memory", "16G")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.kryoserializer.buffer.max", "1000M")
      .config("spark.driver.maxResultSize", "0")

    if(gpu & spark23){
      build.config("spark.jars.packages", MavenGpuSpark23)
    } else if(spark23){
      build.config("spark.jars.packages", MavenSpark23)
    } else if(gpu){
      build.config("spark.jars.packages", MavenGpuSpark24)
    } else {
      build.config("spark.jars.packages", MavenSpark24)
    }

    build.getOrCreate()
  }

  def version(): String = {
    currentVersion
  }

}
