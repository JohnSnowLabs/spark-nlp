from pyspark.sql import SparkSession

import os


class SparkContextForTest:
    spark = SparkSession.builder \
        .master("local[4]") \
        .config("spark.jars", 'lib/sparknlp.jar') \
        .getOrCreate()
    data = spark. \
        read \
        .parquet("file:///" + os.getcwd() + "/../src/test/resources/sentiment.parquet") \
        .limit(100)
    data.cache()
    data.count()


class SparkContextForNER:
    spark = SparkSession.builder \
        .master("local[4]") \
        .config("spark.jars", 'lib/sparknlp.jar') \
        .getOrCreate()
    data = spark. \
        read \
        .csv("file:///" + os.getcwd() + "/../src/test/resources/ner-corpus/icdtest.txt") \
        .limit(100)
    data.cache()
    data.count()
