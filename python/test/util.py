from pyspark.sql import SparkSession

import os


class SparkContextForTest:
    spark = SparkSession.builder \
        .master("local[4]") \
        .config("spark.jars", 'lib/sparknlp.jar,lib/sparknlp-ocr.jar') \
        .config("spark.driver.memory", "4G") \
        .getOrCreate()


class DataForTest:

    data = SparkContextForTest.spark. \
        read \
        .parquet("file:///" + os.getcwd() + "/../src/test/resources/sentiment.parquet") \
        .limit(100)
    data.cache()
    data.count()

    data_ner = SparkContextForTest.spark. \
        read \
        .csv("file:///" + os.getcwd() + "/../src/test/resources/ner-corpus/icdtest.txt") \
        .limit(100)
    data.cache()
    data.count()

    data_tdp = SparkContextForTest.spark.\
        sparkContext.parallelize([["I saw a girl with a telescope"]]).toDF().toDF("text")


