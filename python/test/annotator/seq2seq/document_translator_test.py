#  Copyright 2017-2025 John Snow Labs
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import os
import unittest

import pytest

from sparknlp.annotator import *
from sparknlp.base import *
from test.util import SparkContextForTest

HTML_DIRECTORY = os.getcwd() + "/../src/test/resources/reader/html"
PDF_DIRECTORY = os.getcwd() + "/../src/test/resources/reader/pdf"
TXT_DIRECTORY = os.getcwd() + "/../src/test/resources/reader/txt"
MIXED_DIRECTORY  = os.getcwd() + "/../src/test/resources/reader/mixed"


@pytest.mark.slow
class DocumentTranslatorHtmlTestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkContextForTest.spark

    def test_html(self):
        document_translator = (
            DocumentTranslator.pretrained()
            .setContentType("text/html")
            .setContentPath(f"{HTML_DIRECTORY}/fake-html.html")
            .setOutputCol("translation")
            .setMaxSentenceLength(1500)
            .setMinSentenceLength(800)
            .setNPredict(-1)
            .setNCtx(14000)
            .setBatchSize(4)
            .setNGpuLayers(0)
            .setSrcLang("English")
            .setTgtLang("French")
        )

        pipeline = Pipeline().setStages([document_translator])

        empty_data = self.spark.createDataFrame([[""]]).toDF("text")
        result_df = pipeline.fit(empty_data).transform(empty_data)
        result_df.show(truncate=False)

        self.assertEqual(result_df.count(), 1)
        translations = result_df.select("translation.result").head()[0]
        self.assertTrue(len(translations) > 0)



    def test_pdf(self):
        document_translator = (
            DocumentTranslator.pretrained()
            .setContentType("application/pdf")
            .setContentPath(f"{PDF_DIRECTORY}/pdf-title.pdf")
            .setOutputCol("translation")
            .setMaxSentenceLength(1500)
            .setMinSentenceLength(800)
            .setNPredict(-1)
            .setNCtx(14000)
            .setBatchSize(4)
            .setNGpuLayers(0)
            .setSrcLang("English")
            .setTgtLang("French")
        )

        pipeline = Pipeline().setStages([document_translator])

        empty_data = self.spark.createDataFrame([[""]]).toDF("text")
        result_df = pipeline.fit(empty_data).transform(empty_data)
        result_df.show(truncate=False)

        self.assertEqual(result_df.count(), 1)
        translations = result_df.select("translation.result").head()[0]
        self.assertTrue(len(translations) > 0)



    def test_txt(self):
        document_translator = (
            DocumentTranslator.pretrained()
            .setContentType("text/plain")
            .setContentPath(f"{TXT_DIRECTORY}/long-text.txt")
            .setOutputCol("translation")
            .setMaxSentenceLength(1500)
            .setMinSentenceLength(800)
            .setNPredict(-1)
            .setNCtx(14000)
            .setBatchSize(4)
            .setNGpuLayers(0)
            .setSrcLang("English")
            .setTgtLang("French")
        )

        pipeline = Pipeline().setStages([document_translator])

        empty_data = self.spark.createDataFrame([[""]]).toDF("text")
        result_df = pipeline.fit(empty_data).transform(empty_data)
        result_df.show(truncate=False)

        self.assertEqual(result_df.count(), 1)
        translations = result_df.select("translation.result").head()[0]
        self.assertTrue(len(translations) > 0)


    def test_multiple_documents(self):
        document_translator = (
            DocumentTranslator.pretrained()
            .setContentPath(f"{MIXED_DIRECTORY}/")
            .setOutputCol("translation")
            .setMaxSentenceLength(1500)
            .setMinSentenceLength(800)
            .setNPredict(-1)
            .setNCtx(14000)
            .setBatchSize(4)
            .setNGpuLayers(0)
            .setSrcLang("English")
            .setTgtLang("French")
        )

        pipeline = Pipeline().setStages([document_translator])

        empty_data = self.spark.createDataFrame([[""]]).toDF("text")
        result_df = pipeline.fit(empty_data).transform(empty_data)
        result_df.show(truncate=False)

