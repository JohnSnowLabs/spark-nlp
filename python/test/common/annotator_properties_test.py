import unittest

import pytest

from sparknlp.annotator import *
from sparknlp.base import *
from test.util import SparkContextForTest


@pytest.mark.fast
class AnnotatorPropertiesTest(unittest.TestCase):

    def setUp(self):
        self.title = "Neopets - The gaming website for all your needs."
        self.desc = "Peter Pipers employees are picking pecks of pickled peppers"
        self.data = SparkContextForTest.spark \
            .createDataFrame([[self.title, self.desc]]).toDF("title", "description")

    def runTest(self):

        document_assembler_title = DocumentAssembler() \
            .setInputCol("title") \
            .setOutputCol("document_title")

        document_assembler_desc = DocumentAssembler() \
            .setInputCol("description") \
            .setOutputCol("document_desc")

        sentence_title = SentenceDetector() \
            .setInputCols(["document_title"]) \
            .setOutputCol("sentence_title")

        sentence_desc = SentenceDetector() \
            .setInputCols("document_desc") \
            .setOutputCol("sentence_desc")

        pipeline = Pipeline(stages=[
            document_assembler_title,
            document_assembler_desc,
            sentence_title,
            sentence_desc,
        ])

        result_df = pipeline.fit(self.data).transform(self.data)

        sentence_title_output = result_df.select("sentence_title.result").collect()[0][0][0]
        self.assertEqual(sentence_title_output, self.title)

        sentence_desc_output = result_df.select("sentence_desc.result").collect()[0][0][0]
        self.assertEqual(sentence_desc_output, self.desc)

@pytest.mark.fast
class AnnotatorPropertiesTestInvalidNumberOfColumns(unittest.TestCase):

    def setUp(self):
        self.array_input = ["document_desc", "document_title"]
        self.tuple_input = ("document_desc", "document_title")

    def runTest(self):

        self.assertRaises(TypeError, SentenceDetector().setInputCols, self.array_input)

        self.assertRaises(TypeError, SentenceDetector().setInputCols, self.tuple_input)


@pytest.mark.fast
class AnnotatorPropertiesTestInvalidInput(unittest.TestCase):

    def setUp(self):
        self.dict_input = {"document_desc": "document_title"}
        self.int_input = 4

    def runTest(self):

        self.assertRaises(TypeError, SentenceDetector().setInputCols, self.dict_input)

        self.assertRaises(TypeError, SentenceDetector().setInputCols, self.int_input)
