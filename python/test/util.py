from pyspark.sql import SparkSession

import os


class SparkContextForTest:
    spark = SparkSession.builder \
        .master("local[4]") \
        .config("spark.jars", 'lib/sparknlp.jar,lib/sparknlp-ocr.jar') \
        .config("spark.driver.memory", "3500M") \
        .getOrCreate()
    data = spark. \
        read \
        .parquet("file:///" + os.getcwd() + "/../src/test/resources/sentiment.parquet") \
        .limit(100)
    data.cache()
    data.count()
