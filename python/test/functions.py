import unittest

from pyspark.sql import DataFrame

from sparknlp.annotator import *
from sparknlp.base import *
from sparknlp.functions import map_annotations_cols, map_annotations_col
from test.util import SparkContextForTest


class FunctionMapColumnsTestSpec(unittest.TestCase):

    def runTest(self):
        data = SparkContextForTest.spark.createDataFrame([["Pepito clavo un clavillo", "Tres tristes tigres"],
                                                          ["Un clavillo muy pillo. Que clavillo clavo pablito.",
                                                           "Comian trigo en un trigal"]]).toDF("text", "text2")
        documentAssembler = DocumentAssembler() \
            .setInputCol('text') \
            .setOutputCol('document')
        documentAssembler2 = DocumentAssembler() \
            .setInputCol('text2') \
            .setOutputCol('document2')

        df = documentAssembler.transform(data)
        df = documentAssembler2.transform(df)

        mapped = map_annotations_cols(df.select("document", "document2"),
                                       lambda x: [a.copy(a.result.lower()) for a in x], ["document", "document2"],
                                       "text_tail", "document")
        sentence_detector_dl = SentenceDetector().setInputCols(["text_tail"]).setOutputCol("sentence")
        mapped_sentence = sentence_detector_dl.transform(mapped)
        mapped_sentence.show(truncate=False)



class FunctionMapColumnTestSpec(unittest.TestCase):

    def runTest(self):
        data = SparkContextForTest.spark.createDataFrame([["Pepito clavo un clavillo"],
                                                          ["Un clavillo muy pillo."]]).toDF("text")
        documentAssembler = DocumentAssembler() \
            .setInputCol('text') \
            .setOutputCol('document')

        df = documentAssembler.transform(data)

        mapped = map_annotations_col(df.select("document"),
                                       lambda x: [a.copy(a.result.lower()) for a in x], "document",
                                       "text_tail", "document")
        sentence_detector_dl = SentenceDetector().setInputCols(["text_tail"]).setOutputCol("sentence")
        mapped_sentence = sentence_detector_dl.transform(mapped)
        mapped_sentence.show(truncate=False)