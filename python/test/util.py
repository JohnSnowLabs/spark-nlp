from pyspark.sql import SparkSession

import os

class SparkSessionForTest:
    spark = SparkSession.builder \
        .master("local[*]") \
        .config("spark.jars", 'lib/sparknlp.jar') \
        .config("spark.driver.memory", "12G") \
        .config("spark.driver.maxResultSize", "2G") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.kryoserializer.buffer.max", "500m") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

class SparkContextForTest:
    spark = SparkSessionForTest.spark
    data = spark. \
        read \
        .parquet("file:///" + os.getcwd() + "/../src/test/resources/sentiment.parquet") \
        .limit(100)
    data.cache()
    data.count()

