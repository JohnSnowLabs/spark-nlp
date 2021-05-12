#  Licensed to the Apache Software Foundation (ASF) under one or more
#  contributor license agreements.  See the NOTICE file distributed with
#  this work for additional information regarding copyright ownership.
#  The ASF licenses this file to You under the Apache License, Version 2.0
#  (the "License"); you may not use this file except in compliance with
#  the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import sys
from pyspark.sql import SparkSession
from sparknlp import annotator
from sparknlp.base import DocumentAssembler, Finisher, EmbeddingsFinisher, TokenAssembler, Chunk2Doc, Doc2Chunk

sys.modules['com.johnsnowlabs.nlp.annotators'] = annotator
sys.modules['com.johnsnowlabs.nlp.annotators.tokenizer'] = annotator
sys.modules['com.johnsnowlabs.nlp.annotators.tokenizer.wordpiece'] = annotator
sys.modules['com.johnsnowlabs.nlp.annotators.ner'] = annotator
sys.modules['com.johnsnowlabs.nlp.annotators.ner.regex'] = annotator
sys.modules['com.johnsnowlabs.nlp.annotators.ner.crf'] = annotator
sys.modules['com.johnsnowlabs.nlp.annotators.ner.dl'] = annotator
sys.modules['com.johnsnowlabs.nlp.annotators.pos'] = annotator
sys.modules['com.johnsnowlabs.nlp.annotators.pos.perceptron'] = annotator
sys.modules['com.johnsnowlabs.nlp.annotators.sbd'] = annotator
sys.modules['com.johnsnowlabs.nlp.annotators.sbd.pragmatic'] = annotator
sys.modules['com.johnsnowlabs.nlp.annotators.sbd.deep'] = annotator
sys.modules['com.johnsnowlabs.nlp.annotators.sda'] = annotator
sys.modules['com.johnsnowlabs.nlp.annotators.sda.pragmatic'] = annotator
sys.modules['com.johnsnowlabs.nlp.annotators.sda.vivekn'] = annotator
sys.modules['com.johnsnowlabs.nlp.annotators.spell'] = annotator
sys.modules['com.johnsnowlabs.nlp.annotators.spell.norvig'] = annotator
sys.modules['com.johnsnowlabs.nlp.annotators.spell.symmetric'] = annotator
sys.modules['com.johnsnowlabs.nlp.annotators.parser'] = annotator
sys.modules['com.johnsnowlabs.nlp.annotators.parser.dep'] = annotator
sys.modules['com.johnsnowlabs.nlp.annotators.parser.typdep'] = annotator
sys.modules['com.johnsnowlabs.nlp.embeddings'] = annotator
sys.modules['com.johnsnowlabs.nlp.annotators.classifier'] = annotator
sys.modules['com.johnsnowlabs.nlp.annotators.classifier.dl'] = annotator
sys.modules['com.johnsnowlabs.nlp.annotators.spell.context'] = annotator
sys.modules['com.johnsnowlabs.nlp.annotators.ld'] = annotator
sys.modules['com.johnsnowlabs.nlp.annotators.ld.dl'] = annotator
sys.modules['com.johnsnowlabs.nlp.annotators.sentence_detector_dl'] = annotator
sys.modules['com.johnsnowlabs.nlp.annotators.seq2seq'] = annotator
sys.modules['com.johnsnowlabs.nlp.annotators.ws'] = annotator

annotators = annotator
embeddings = annotator


def start(gpu=False, spark23=False, spark24=False, memory="16G"):
    current_version = "3.0.3"

    # Spark NLP on Apache Spark 3.0.x
    maven_spark = "com.johnsnowlabs.nlp:spark-nlp_2.12:{}".format(current_version)
    maven_gpu_spark = "com.johnsnowlabs.nlp:spark-nlp-gpu_2.12:{}".format(current_version)
    # Spark NLP on Apache Spark 2.4.x
    maven_spark24 = "com.johnsnowlabs.nlp:spark-nlp-spark24_2.11:{}".format(current_version)
    maven_gpu_spark24 = "com.johnsnowlabs.nlp:spark-nlp-gpu-spark24_2.11:{}".format(current_version)
    # Spark NLP on Apache Spark 2.3.x
    maven_spark23 = "com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:{}".format(current_version)
    maven_gpu_spark23 = "com.johnsnowlabs.nlp:spark-nlp-gpu-spark23_2.11:{}".format(current_version)

    builder = SparkSession.builder \
        .appName("Spark NLP") \
        .master("local[*]") \
        .config("spark.driver.memory", memory) \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.kryoserializer.buffer.max", "2000M") \
        .config("spark.driver.maxResultSize", "0")

    if gpu and spark23:
        builder.config("spark.jars.packages", maven_gpu_spark23)
    elif gpu and spark24:
        builder.config("spark.jars.packages", maven_gpu_spark24)
    elif spark23:
        builder.config("spark.jars.packages", maven_spark23)
    elif spark24:
        builder.config("spark.jars.packages", maven_spark24)
    elif gpu:
        builder.config("spark.jars.packages", maven_gpu_spark)
    else:
        builder.config("spark.jars.packages", maven_spark)

    return builder.getOrCreate()


def version():
    return '3.0.3'
