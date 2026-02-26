#  Copyright 2017-2022 John Snow Labs
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os

from pyspark.sql import SparkSession
import time


class SparkSessionForTest:
    jars_path = "lib/sparknlp.jar"
    spark = SparkSession.builder \
        .master("local[*]") \
        .config("spark.jars", jars_path) \
        .config("spark.driver.memory", "12G") \
        .config("spark.driver.maxResultSize", "2G") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.kryoserializer.buffer.max", "500m") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")


class SparkContextForTest:
    parquet_file = "file:///" + os.getcwd() + "/../src/test/resources/sentiment.parquet"
    spark = SparkSessionForTest.spark
    data = spark. \
        read \
        .parquet(parquet_file) \
        .limit(100)
    data.cache()
    data.count()


def getFreeRAM():
    """Returns the amount of free system RAM in Megabytes (MB)."""
    import psutil # Not available no github runners
    return psutil.virtual_memory().free / (1024 * 1024)


def measureRAMChange(func, *args, **kwargs):
    ramBefore = getFreeRAM()
    func(*args, **kwargs)
    time.sleep(0.5)
    ramAfter = getFreeRAM()
    return ramAfter - ramBefore
