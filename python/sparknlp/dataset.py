from sparknlp.internal import ExtendedJavaWrapper
from sparknlp.common import ExternalResource, ReadAs
from pyspark.sql import SparkSession, DataFrame


class CoNLL(ExtendedJavaWrapper):
    def __init__(self,
                 targetColumn = 3,
                 posColumn = 1,
                 textColumn = "text",
                 docColumn = "document",
                 sentenceColumn = "sentence",
                 tokenColumn = "token",
                 posTaggedColumn = "pos",
                 labelColumn = "label"):
        super(CoNLL, self).__init__("com.johnsnowlabs.nlp.datasets.CoNLL")

        self._java_obj = self._new_java_obj(self._java_obj,
                                            targetColumn,
                                            posColumn,
                                            textColumn,
                                            docColumn,
                                            sentenceColumn,
                                            tokenColumn,
                                            posTaggedColumn,
                                            labelColumn
                                            )

    def readDataset(self, path, read_as=ReadAs.LINE_BY_LINE, opts={}):
        resource = ExternalResource(path, read_as, opts)

        session = SparkSession(self.sc)
        jSession = session._jsparkSession

        jdf = self._java_obj.readDataset(resource, jSession)
        return DataFrame(jdf, session._wrapped)


