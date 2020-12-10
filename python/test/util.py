from pyspark.sql import SparkSession

import os


class SparkContextForTest:
    spark = SparkSession.builder \
        .master("local[4]") \
        .config("spark.jars", 'lib/sparknlp.jar') \
        .config("spark.driver.memory", "6500M") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    data = spark. \
        read \
        .parquet("file:///" + os.getcwd() + "/../src/test/resources/sentiment.parquet") \
        .limit(100)
    data.cache()
    data.count()


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

