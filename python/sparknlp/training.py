from sparknlp.internal import ExtendedJavaWrapper
from sparknlp.common import ExternalResource, ReadAs
from pyspark.sql import SparkSession, DataFrame


class CoNLL(ExtendedJavaWrapper):
    def __init__(self,
                 documentCol = 'document',
                 sentenceCol = 'sentence',
                 tokenCol = 'token',
                 posCol = 'pos',
                 conllLabelIndex = 3,
                 conllPosIndex = 1,
                 textCol = 'text',
                 labelCol = 'label',
                 explodeSentences = True,
                 ):
        super(CoNLL, self).__init__("com.johnsnowlabs.nlp.training.CoNLL",
                                    documentCol,
                                    sentenceCol,
                                    tokenCol,
                                    posCol,
                                    conllLabelIndex,
                                    conllPosIndex,
                                    textCol,
                                    labelCol,
                                    explodeSentences)

    def readDataset(self, spark, path, read_as=ReadAs.LINE_BY_LINE):

        # ToDo Replace with std pyspark
        jSession = spark._jsparkSession

        jdf = self._java_obj.readDataset(jSession, path, read_as)
        return DataFrame(jdf, spark._wrapped)


class POS(ExtendedJavaWrapper):
    def __init__(self):
        super(POS, self).__init__("com.johnsnowlabs.nlp.training.POS")

    def readDataset(self, spark, path, delimiter="|", outputPosCol="tags", outputDocumentCol="document", outputTextCol="text"):

        # ToDo Replace with std pyspark
        jSession = spark._jsparkSession

        jdf = self._java_obj.readDataset(jSession, path, delimiter, outputPosCol, outputDocumentCol, outputTextCol)
        return DataFrame(jdf, spark._wrapped)
