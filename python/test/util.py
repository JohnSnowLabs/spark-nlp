from pyspark.sql import SparkSession


class SparkContextForTest:
    spark = SparkSession.builder \
        .master("local[4]") \
        .config("spark.jars", 'lib/sparknlp.jar') \
        .getOrCreate()
    data = spark. \
        read \
        .parquet("../src/test/resources/sentiment.parquet") \
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
        .csv("../src/test/resources/ner-corpus/icdtest.txt") \
        .limit(100)
    data.cache()
    data.count()
