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

def start(gpu=False, spark23=False):
    current_version = "2.7.4"

    maven_spark24 = "com.johnsnowlabs.nlp:spark-nlp_2.11:{}".format(current_version)
    maven_gpu_spark24 = "com.johnsnowlabs.nlp:spark-nlp-gpu_2.11:{}".format(current_version)
    maven_spark23 = "com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:{}".format(current_version)
    maven_gpu_spark23 = "com.johnsnowlabs.nlp:spark-nlp-gpu-spark23_2.11:{}".format(current_version)

    builder = SparkSession.builder \
        .appName("Spark NLP") \
        .master("local[*]") \
        .config("spark.driver.memory", "16G") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.kryoserializer.buffer.max", "1000M") \
        .config("spark.driver.maxResultSize", "0")

    if gpu and spark23:
        builder.config("spark.jars.packages", maven_gpu_spark23)
    elif spark23:
        builder.config("spark.jars.packages", maven_spark23)
    elif gpu:
        builder.config("spark.jars.packages", maven_gpu_spark24)
    else:
        builder.config("spark.jars.packages", maven_spark24)
        
    return builder.getOrCreate()


def version():
    return '2.7.4'
