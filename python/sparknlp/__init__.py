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
import subprocess
import threading
from pyspark.sql import SparkSession
from sparknlp import annotator
from sparknlp.base import DocumentAssembler, Finisher, EmbeddingsFinisher, TokenAssembler, Chunk2Doc, Doc2Chunk
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.java_gateway import launch_gateway

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


def start(gpu=False, spark23=False, spark24=False, memory="16G", real_time_output=False, output_level=1):

    current_version = "3.1.1"

    class SparkNLPConfig:

        def __init__(self):
            self.master, self.app_name = "local[*]", "Spark NLP"
            self.serializer, self.serializer_max_buffer = "org.apache.spark.serializer.KryoSerializer", "2000M"
            self.driver_max_result_size = "0"
            # Spark NLP on Apache Spark 3.0.x
            self.maven_spark = "com.johnsnowlabs.nlp:spark-nlp_2.12:{}".format(current_version)
            self.maven_gpu_spark = "com.johnsnowlabs.nlp:spark-nlp-gpu_2.12:{}".format(current_version)
            # Spark NLP on Apache Spark 2.4.x
            self.maven_spark24 = "com.johnsnowlabs.nlp:spark-nlp-spark24_2.11:{}".format(current_version)
            self.maven_gpu_spark24 = "com.johnsnowlabs.nlp:spark-nlp-gpu-spark24_2.11:{}".format(current_version)
            # Spark NLP on Apache Spark 2.3.x
            self.maven_spark23 = "com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:{}".format(current_version)
            self.maven_gpu_spark23 = "com.johnsnowlabs.nlp:spark-nlp-gpu-spark23_2.11:{}".format(current_version)

    def start_without_realtime_output():
        builder = SparkSession.builder \
            .appName(spark_nlp_config.app_name) \
            .master(spark_nlp_config.master) \
            .config("spark.driver.memory", memory) \
            .config("spark.serializer", spark_nlp_config.serializer) \
            .config("spark.kryoserializer.buffer.max", spark_nlp_config.serializer_max_buffer) \
            .config("spark.driver.maxResultSize", spark_nlp_config.driver_max_result_size)

        if gpu and spark23:
            builder.config("spark.jars.packages", spark_nlp_config.maven_gpu_spark23)
        elif gpu and spark24:
            builder.config("spark.jars.packages", spark_nlp_config.maven_gpu_spark24)
        elif spark23:
            builder.config("spark.jars.packages", spark_nlp_config.maven_spark23)
        elif spark24:
            builder.config("spark.jars.packages", spark_nlp_config.maven_spark24)
        elif gpu:
            builder.config("spark.jars.packages", spark_nlp_config.maven_gpu_spark)
        else:
            builder.config("spark.jars.packages", spark_nlp_config.maven_spark)

        return builder.getOrCreate()

    def start_with_realtime_output():

        class SparkWithCustomGateway:

            def __init__(self):
                spark_conf = SparkConf()
                spark_conf.setAppName(spark_nlp_config.app_name)
                spark_conf.setMaster(spark_nlp_config.master)
                spark_conf.set("spark.driver.memory", memory)
                spark_conf.set("spark.serializer", spark_nlp_config.serializer)
                spark_conf.set("spark.kryoserializer.buffer.max", spark_nlp_config.serializer_max_buffer)
                spark_conf.set("spark.driver.maxResultSize", spark_nlp_config.driver_max_result_size)

                if gpu:
                    spark_conf.set("spark.jars.packages", spark_nlp_config.maven_gpu_spark)
                else:
                    spark_conf.set("spark.jars.packages", spark_nlp_config.maven_spark)

                # Make the py4j JVM stdout and stderr available without buffering
                popen_kwargs = {
                    'stdout': subprocess.PIPE,
                    'stderr': subprocess.PIPE,
                    'bufsize': 0
                }

                # Launch the gateway with our custom settings
                self.gateway = launch_gateway(conf=spark_conf, popen_kwargs=popen_kwargs)
                self.process = self.gateway.proc
                # Use the gateway we launched
                spark_context = SparkContext(gateway=self.gateway)
                self.spark_session = SparkSession(spark_context)

                self.out_thread = threading.Thread(target=self.output_reader)
                self.error_thread = threading.Thread(target=self.error_reader)
                self.std_background_listeners()

            def std_background_listeners(self):
                self.out_thread.start()
                self.error_thread.start()

            def output_reader(self):
                for line in iter(self.process.stdout.readline, b''):
                    print('{0}'.format(line.decode('utf-8')), end='')

            def error_reader(self):
                RED = '\033[91m'
                RESET = '\033[0m'
                for line in iter(self.process.stderr.readline, b''):
                    if output_level == 0:
                        print(RED + '{0}'.format(line.decode('utf-8')) + RESET, end='')
                    else:
                        # output just info
                        pass

            def shutdown(self):
                self.spark_session.stop()
                self.gateway.shutdown()
                self.process.communicate()

                self.out_thread.join()
                self.error_thread.join()

        return SparkWithCustomGateway()

    spark_nlp_config = SparkNLPConfig()

    if real_time_output:
        if spark23 or spark24:
            spark_session = start_without_realtime_output()
            return spark_session
        else:
            # Available from Spark 3.0.x
            class SparkRealTimeOutput:

                def __init__(self):
                    self.__spark_with_custom_gateway = start_with_realtime_output()
                    self.spark_session = self.__spark_with_custom_gateway.spark_session

                def shutdown(self):
                    self.__spark_with_custom_gateway.shutdown()

            return SparkRealTimeOutput()
    else:
        spark_session = start_without_realtime_output()
        return spark_session


def version():
    return '3.1.1'
