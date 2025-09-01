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

import subprocess
import sys
import threading

from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.java_gateway import launch_gateway
from pyspark.sql import SparkSession

from sparknlp import annotator
# Must be declared here one by one or else PretrainedPipeline will fail with AttributeError
from sparknlp.base import DocumentAssembler, MultiDocumentAssembler, Finisher, EmbeddingsFinisher, TokenAssembler, \
    Doc2Chunk, AudioAssembler, GraphFinisher, ImageAssembler, TableAssembler
from sparknlp.reader import SparkNLPReader

sys.modules['com.johnsnowlabs.nlp.annotators'] = annotator
sys.modules['com.johnsnsowlabs.nlp.annotators.tokenizer'] = annotator
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
sys.modules['com.johnsnowlabs.nlp.annotators.er'] = annotator
sys.modules['com.johnsnowlabs.nlp.annotators.coref'] = annotator
sys.modules['com.johnsnowlabs.nlp.annotators.cv'] = annotator
sys.modules['com.johnsnowlabs.nlp.annotators.audio'] = annotator
sys.modules['com.johnsnowlabs.ml.ai'] = annotator

annotators = annotator
embeddings = annotator

__version__ = "6.1.3"


def start(gpu=False,
          apple_silicon=False,
          aarch64=False,
          memory="16G",
          cache_folder="",
          log_folder="",
          cluster_tmp_dir="",
          params=None,
          real_time_output=False,
          output_level=1):
    """Starts a PySpark instance with default parameters for Spark NLP.

    The default parameters would result in the equivalent of:

    .. code-block:: python

        SparkSession.builder \\
            .appName("Spark NLP") \\
            .master("local[*]") \\
            .config("spark.driver.memory", "16G") \\
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \\
            .config("spark.kryoserializer.buffer.max", "2000M") \\
            .config("spark.driver.maxResultSize", "0") \\
            .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:|release|") \\
            .getOrCreate()

    Parameters
    ----------
    gpu : bool, optional
        Whether to enable GPU acceleration (must be set up correctly), by default False
    apple_silicon : bool, optional
        Whether to enable Apple Silicon support for macOS
    aarch64 : bool, optional
        Whether to enable Linux Aarch64 support
    memory : str, optional
        How much memory to allocate for the Spark driver, by default "16G"
    cache_folder : str, optional
        The location to download and extract pretrained Models and Pipelines. If not
        set, it will be in the users home directory under `cache_pretrained`.
    log_folder : str, optional
        The location to use on a cluster for temporarily files such as unpacking indexes
        for WordEmbeddings. By default, this locations is the location of
        `hadoop.tmp.dir` set via Hadoop configuration for Apache Spark. NOTE: `S3` is
        not supported and it must be local, HDFS, or DBFS.
    params : dict, optional
        Custom parameters to set for the Spark configuration, by default None.
    cluster_tmp_dir : str, optional
        The location to save logs from annotators during training. If not set, it will
        be in the users home directory under `annotator_logs`.
    real_time_output : bool, optional
        Whether to read and print JVM output in real time, by default False
    output_level : int, optional
        Output level for logs, by default 1

    Notes
    -----
    Since Spark version 3.2, Python 3.6 is deprecated. If you are using this
    python version, consider sticking to lower versions of Spark.

    Returns
    -------
    :class:`SparkSession`
        The initiated Spark session.

    """
    current_version = __version__

    if params is None:
        params = {}
    else:
        if not isinstance(params, dict):
            raise TypeError('params must be a dictionary like {"spark.executor.memory": "8G"}')

    if '_instantiatedSession' in dir(SparkSession) and SparkSession._instantiatedSession is not None:
        print('Warning::Spark Session already created, some configs may not take.')

    driver_cores = "*"
    for key, value in params.items():
        if key == "spark.driver.cores":
            driver_cores = f"{value}"
        else:
            driver_cores = "*"

    class SparkNLPConfig:

        def __init__(self):
            self.master, self.app_name = "local[{}]".format(driver_cores), "Spark NLP"
            self.serializer, self.serializer_max_buffer = "org.apache.spark.serializer.KryoSerializer", "2000M"
            self.driver_max_result_size = "0"
            # Spark NLP on CPU or GPU
            self.maven_spark3 = "com.johnsnowlabs.nlp:spark-nlp_2.12:{}".format(current_version)
            self.maven_gpu_spark3 = "com.johnsnowlabs.nlp:spark-nlp-gpu_2.12:{}".format(current_version)
            # Spark NLP on Apple Silicon
            self.maven_silicon = "com.johnsnowlabs.nlp:spark-nlp-silicon_2.12:{}".format(current_version)
            # Spark NLP on Linux Aarch64
            self.maven_aarch64 = "com.johnsnowlabs.nlp:spark-nlp-aarch64_2.12:{}".format(current_version)

    def start_without_realtime_output():
        builder = SparkSession.builder \
            .appName(spark_nlp_config.app_name) \
            .master(spark_nlp_config.master) \
            .config("spark.driver.memory", memory) \
            .config("spark.serializer", spark_nlp_config.serializer) \
            .config("spark.kryoserializer.buffer.max", spark_nlp_config.serializer_max_buffer) \
            .config("spark.driver.maxResultSize", spark_nlp_config.driver_max_result_size)

        if apple_silicon:
            spark_jars_packages = spark_nlp_config.maven_silicon
        elif aarch64:
            spark_jars_packages = spark_nlp_config.maven_aarch64
        elif gpu:
            spark_jars_packages = spark_nlp_config.maven_gpu_spark3
        else:
            spark_jars_packages = spark_nlp_config.maven_spark3

        if cache_folder != '':
            builder.config("spark.jsl.settings.pretrained.cache_folder", cache_folder)
        if log_folder != '':
            builder.config("spark.jsl.settings.annotator.log_folder", log_folder)
        if cluster_tmp_dir != '':
            builder.config("spark.jsl.settings.storage.cluster_tmp_dir", cluster_tmp_dir)

        if params.get("spark.jars.packages") is None:
            builder.config("spark.jars.packages", spark_jars_packages)

        for key, value in params.items():
            if key == "spark.jars.packages":
                packages = spark_jars_packages + "," + value
                builder.config(key, packages)
            else:
                builder.config(key, value)

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

                if apple_silicon:
                    spark_jars_packages = spark_nlp_config.maven_silicon
                elif aarch64:
                    spark_jars_packages = spark_nlp_config.maven_aarch64
                elif gpu:
                    spark_jars_packages = spark_nlp_config.maven_gpu_spark3
                else:
                    spark_jars_packages = spark_nlp_config.maven_spark3

                if cache_folder != '':
                    spark_conf.set("spark.jsl.settings.pretrained.cache_folder", cache_folder)
                if log_folder != '':
                    spark_conf.set("spark.jsl.settings.annotator.log_folder", log_folder)
                if cluster_tmp_dir != '':
                    spark_conf.set("spark.jsl.settings.storage.cluster_tmp_dir", cluster_tmp_dir)

                if params.get("spark.jars.packages") is None:
                    spark_conf.set("spark.jars.packages", spark_jars_packages)

                for key, value in params.items():
                    if key == "spark.jars.packages":
                        packages = spark_jars_packages + "," + value
                        spark_conf.set(key, packages)
                    else:
                        spark_conf.set(key, value)

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
        # Available from Spark 3.0.x
        class SparkRealTimeOutput:

            def __init__(self):
                self.__spark_with_custom_gateway = start_with_realtime_output()
                self.spark_session = self.__spark_with_custom_gateway.spark_session

            def shutdown(self):
                self.__spark_with_custom_gateway.shutdown()

        return SparkRealTimeOutput().spark_session
    else:
        spark_session = start_without_realtime_output()
        return spark_session

def read(params=None):
    spark_session = start()
    return SparkNLPReader(spark_session, params)

def version():
    """Returns the current Spark NLP version.

    Returns
    -------
    str
        The current Spark NLP version.
    """
    return __version__
