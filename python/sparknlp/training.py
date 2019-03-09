from sparknlp.internal import ExtendedJavaWrapper
from sparknlp.common import ExternalResource, ReadAs
from pyspark.sql import SparkSession, DataFrame


class CoNLL(ExtendedJavaWrapper):
    def __init__(self,
                 documentCol,
                 sentenceCol,
                 tokenCol,
                 posCol,
                 conllLabelIndex = 3,
                 conllPosIndex = 1,
                 textCol = "text",
                 labelCol = "label"):
        super(CoNLL, self).__init__("com.johnsnowlabs.nlp.training.CoNLL")

        self._java_obj = self._new_java_obj(self._java_obj,
                                            documentCol,
                                            sentenceCol,
                                            tokenCol,
                                            posCol,
                                            conllLabelIndex,
                                            conllPosIndex,
                                            textCol,
                                            labelCol
                                            )

    def readDataset(self, path, read_as=ReadAs.LINE_BY_LINE, opts={}):
        resource = ExternalResource(path, read_as, opts)

        # ToDo Replace with std pyspark
        session = SparkSession(self.sc)
        jSession = session._jsparkSession

        jdf = self._java_obj.readDataset(resource, jSession)
        return DataFrame(jdf, session._wrapped)


