import sparknlp.internal as _internal


class Embeddings:
    def __init__(self, embeddings):
        self.jembeddings = embeddings


class EmbeddingsHelper:
    @classmethod
    def load(cls, path, spark_session, embeddings_format, embeddings_dim, embeddings_casesens=False):
        jembeddings = _internal._EmbeddingsHelperLoad(path, spark_session, embeddings_format, embeddings_dim, embeddings_casesens).apply()
        return Embeddings(jembeddings)

    @classmethod
    def save(cls, path, embeddings, spark_session):
        return _internal._EmbeddingsHelperSave(path, embeddings, spark_session).apply()

    @classmethod
    def clearCache(cls):
        return _internal._EmbeddingsHelperClear().apply()

    @classmethod
    def getFromAnnotator(cls, annotator):
        return _internal._EmbeddingsHelperFromAnnotator(annotator).apply()

    @classmethod
    def getByRef(cls, ref):
        return _internal._EmbeddingsHelperByRef(ref).apply()

    @classmethod
    def setRef(cls, ref, embeddings):
        return _internal._EmbeddingsHelperSetRef(ref, embeddings).apply()
