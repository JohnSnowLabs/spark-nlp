import sparknlp.internal as _internal

from pyspark.ml.param import Params
from pyspark import keyword_only
import sys
import threading
import time
import sparknlp.pretrained as _pretrained


# DONT REMOVE THIS IMPORT
from sparknlp.annotator import WordEmbeddingsModel
####


class RocksDBConnection:
    def __init__(self, connection):
        self.jconnection = connection


class WordEmbeddingsLoader:
    @classmethod
    def load(cls, path, spark_session, embeddings_format, embeddings_ref):
        print("Loading started this may take some time")
        stop_threads = False
        t1 = threading.Thread(target=_pretrained.printProgress, args=(lambda: stop_threads,))
        t1.start()
        jembeddings = _internal._WordEmbeddingsLoader(path, spark_session, embeddings_format, embeddings_ref).apply()
        stop_threads = True
        t1.join()
        print("Loading done")
        return RocksDBConnection(jembeddings)
