import sparknlp.internal as _internal


class Embeddings:
    def __init__(self, embeddings):
        self.jembeddings = embeddings


class EmbeddingsHelper:
    @classmethod
    def loadEmbeddings(cls, path, spark_session, embeddings_format, embeddings_dim, embeddings_casesens=True, place_in_cache=""):
        jembeddings = _internal._EmbeddingsHelperLoad(path, spark_session, embeddings_format, embeddings_dim, embeddings_casesens, place_in_cache).apply()
        return Embeddings(jembeddings)

    @classmethod
    def saveEmbeddings(cls, path, embeddings, spark_session):
        return _internal._EmbeddingsHelperSave(path, embeddings, spark_session).apply()

    @classmethod
    def clearCache(cls):
        return _internal._EmbeddingsHelperClear().apply()

    @classmethod
    def getEmbeddingsFromAnnotator(cls, annotator):
        return _internal._EmbeddingsHelperFromAnnotator(annotator).apply()
