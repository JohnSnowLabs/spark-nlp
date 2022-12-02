import unittest

import pytest
from pyspark.ml import Pipeline

from sparknlp.annotator import *
from test.util import SparkSessionForTest


@pytest.mark.fast
class NerConverterTestSpec(unittest.TestCase):

    def runTest(self):
        documentAssembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token")
        embeddings = WordEmbeddingsModel.pretrained() \
            .setInputCols(["document", "token"]) \
            .setOutputCol("bert")
        nerTagger = NerDLModel.pretrained() \
            .setInputCols(["document", "token", "bert"]) \
            .setOutputCol("ner")
        pipeline = Pipeline().setStages([
            documentAssembler,
            tokenizer,
            embeddings,
            nerTagger
        ])
        data = SparkSessionForTest.spark.createDataFrame(
            [["U.N. official Ekeus heads for Baghdad."]]).toDF("text")
        result = pipeline.fit(data).transform(data)

        converter = NerConverter() \
            .setInputCols(["document", "token", "ner"]) \
            .setOutputCol("entities") \
            .setPreservePosition(False) \
            .setWhiteList(["ORG", "LOC"])

        converter.transform(result).selectExpr("explode(entities)").show(truncate=False)
