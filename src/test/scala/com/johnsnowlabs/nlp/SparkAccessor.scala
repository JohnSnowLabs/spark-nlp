package com.johnsnowlabs.nlp

import org.apache.spark.sql.SparkSession

object SparkAccessor {
  val spark: SparkSession = SparkSession
    .builder()
    .appName("test")
    .master("local[1]")
    .config("spark.driver.memory", "4G")
    .config("spark.kryoserializer.buffer.max","200M")
    .config("spark.serializer","org.apache.spark.serializer.KryoSerializer")
    .config("spark.kryo.registrator", "com.johnsnowlabs.nlp.annotators.spell.context.ContextSpellRegistrator")
    .getOrCreate()


  val benchmarkSpark: SparkSession = SparkSession
    .builder()
    .appName("benchmark")
    .master("local[1]")
    .config("spark.driver.memory", "8G")
    .config("spark.kryoserializer.buffer.max","200M")
    .config("spark.serializer","org.apache.spark.serializer.KryoSerializer")
    .getOrCreate()
}
