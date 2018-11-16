from pyspark.sql import SparkSession

import os


class SparkContextForTest:
    spark = SparkSession.builder \
        .master("local[4]") \
        .config("spark.jars", 'lib/sparknlp.jar,lib/sparknlp-ocr.jar') \
        .getOrCreate()
    data = spark. \
        read \
        .parquet("file:///" + os.getcwd() + "/../../src/test/resources/sentiment.parquet") \
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
        .csv("file:///" + os.getcwd() + "/../../src/test/resources/ner-corpus/icdtest.txt") \
        .limit(100)
    data.cache()
    data.count()


class SparkContextForTDP:
    spark = SparkSession.builder \
        .master("local[1]") \
        .config("spark.driver.memory", "4G") \
        .config("spark.driver.maxResultSize", "1G") \
        .config("spark.executor.memory", "4G") \
        .config("spark.driver.extraJavaOptions", "-Xms1G -Xss1M") \
        .config("spark.executor.extraJavaOptions", "-Xms1G -Xss1M") \
        .getOrCreate()

    data = spark.sparkContext.parallelize([["I saw a girl with a telescope"]]).toDF().toDF("text")

# --conf spark.driver.extraJavaOptions="-Xss10m -XX:MaxPermSize=1024M "
# --conf spark.executor.extraJavaOptions="-Xss10m -XX:MaxPermSize=512M"
# .config("spark.driver.extraJavaOptions=",
#         "-Xms1G -Xmx4G -Xss1M -XX:+CMSClassUnloadingEnabled") \
# .config("spark.executor.extraJavaOptions=",
#         "-Xms1G -Xmx4G -Xss1M -XX:+CMSClassUnloadingEnabled") \
