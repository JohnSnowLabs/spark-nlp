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


class Embeddings:
    def __init__(self, embeddings):
        self.jembeddings = embeddings


class EmbeddingsHelper:
    @classmethod
    def load(cls, path, spark_session, embeddings_format, embeddings_ref, embeddings_dim, embeddings_casesens=False):
        print("Loading started this may take some time")
        stop_threads = False
        t1 = threading.Thread(target=_pretrained.printProgress, args=(lambda: stop_threads,))
        t1.start()
        jembeddings = _internal._EmbeddingsHelperLoad(path, spark_session, embeddings_format, embeddings_ref, embeddings_dim, embeddings_casesens).apply()
        stop_threads = True
        t1.join()
        print("Loading done")
        return Embeddings(jembeddings)

    @classmethod
    def save(cls, path, embeddings, spark_session):
        return _internal._EmbeddingsHelperSave(path, embeddings, spark_session).apply()

    @classmethod
    def getFromAnnotator(cls, annotator):
        return _internal._EmbeddingsHelperFromAnnotator(annotator).apply()
