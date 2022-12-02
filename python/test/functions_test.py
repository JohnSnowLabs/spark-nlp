#  Copyright 2017-2022 John Snow Labs
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

import unittest

import pytest

from sparknlp.annotator import *
from sparknlp.base import *
from sparknlp.functions import map_annotations_cols, map_annotations_col
from test.util import SparkContextForTest


@pytest.mark.fast
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


@pytest.mark.fast
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