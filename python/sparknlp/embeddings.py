import sparknlp.internal as _internal

from sparknlp.common import AnnotatorApproach, AnnotatorModel, HasWordEmbeddings, HasEmbeddings
from sparknlp.internal import _BertLoader

from pyspark.ml.param.shared import Param, TypeConverters
from pyspark.ml.param import Params
from pyspark import keyword_only


class Embeddings:
    def __init__(self, embeddings):
        self.jembeddings = embeddings


class EmbeddingsHelper:
    @classmethod
    def load(cls, path, spark_session, embeddings_format, embeddings_ref, embeddings_dim, embeddings_casesens=False):
        jembeddings = _internal._EmbeddingsHelperLoad(path, spark_session, embeddings_format, embeddings_ref, embeddings_dim, embeddings_casesens).apply()
        return Embeddings(jembeddings)

    @classmethod
    def save(cls, path, embeddings, spark_session):
        return _internal._EmbeddingsHelperSave(path, embeddings, spark_session).apply()

    @classmethod
    def getFromAnnotator(cls, annotator):
        return _internal._EmbeddingsHelperFromAnnotator(annotator).apply()
