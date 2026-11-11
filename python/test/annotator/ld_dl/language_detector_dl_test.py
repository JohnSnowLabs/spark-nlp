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
import os
import unittest

import pytest

from sparknlp.annotator import *
from sparknlp.base import *
from test.util import SparkContextForTest


@pytest.mark.slow
class LanguageDetectorDLTestSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.spark.read \
            .option("delimiter", "|") \
            .option("header", "true") \
            .csv(path="file:///" + os.getcwd() + "/../src/test/resources/language-detector/multilingual_sample.txt")

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        sentence_detector = SentenceDetectorDLModel.pretrained() \
            .setInputCols(["document"]) \
            .setOutputCol("sentence")

        ld = LanguageDetectorDL.pretrained()

        pipeline = Pipeline(stages=[
            document_assembler,
            sentence_detector,
            ld
        ])

        # list all the languages
        print(ld.getLanguages())

        model = pipeline.fit(self.data)
        model.transform(self.data).show()


@pytest.mark.slow
class LanguageDetectorDLGlotLIDTestSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.spark.createDataFrame([
            ["The patient was admitted with chest pain and shortness of breath."],
            ["患者因胸痛和呼吸困难三天入院。"],
            ["El paciente ingresó por dolor torácico y dificultad para respirar."],
            ["No oral lesions."],
            ["血液檢查顯示肌鈣蛋白升高,胸部X光片符合肺水腫表現。"],
        ]).toDF("text")

        self.document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

    def test_run(self):
        ld = LanguageDetectorDL.pretrained("ld_glotlid_v4", "xx") \
            .setInputCols(["document"]) \
            .setOutputCol("language") \
            .setScriptPooling(True) \
            .setMinLetters(20)

        result = Pipeline(stages=[self.document_assembler, ld]).fit(self.data).transform(self.data)
        result.select("text", "language.result").show(truncate=False)

        labels = [row.result[0] for row in result.select("language.result").collect()]
        self.assertEqual(labels, ["eng_Latn", "cmn_Hani", "spa_Latn", "unk", "yue_Hani"])

    def test_params(self):
        ld = LanguageDetectorDL.pretrained("ld_glotlid_v4", "xx") \
            .setInputCols(["document"]) \
            .setOutputCol("language") \
            .setTopK(2) \
            .setMinLetters(0) \
            .setScriptPooling(False)

        result = Pipeline(stages=[self.document_assembler, ld]).fit(self.data).transform(self.data)
        annotations = [row.language[0] for row in result.select("language").collect()]

        self.assertEqual(annotations[3].result, "cat_Latn")
        self.assertEqual(annotations[4].result, "unk")
        for annotation in annotations:
            self.assertEqual(len(annotation.metadata), 4)

