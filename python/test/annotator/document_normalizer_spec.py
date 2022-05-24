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
from test.util import SparkSessionForTest


@pytest.mark.fast
class DocumentNormalizerSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkSessionForTest.spark.createDataFrame([
            ["""<span style="font-weight: bold; font-size: 8pt">
                     <pre style="font-family: verdana">
                      <b>The Output Y(s) of the fig. is:
                       <br /><br />
                       <img src="http://192.168.5.151/UADP4.0/ItemAuthoring/QuestionBank/Resources/94954.jpeg" />
                      </b>
                     </pre>
                    </span>"""],
            ["""<!DOCTYPE html>
                    <html>
                    <body>
                    <a class='w3schools-logo notranslate' href='//www.w3schools.com'>w3schools<span class='dotcom'>.com</span></a>
                    <h1 style="font-size:300%;">This is a heading</h1>
                    <p style="font-size:160%;">This is a paragraph containing some PII like jonhdoe@myemail.com ! John is now 42 years old.</p>
                    <p style="font-size:160%;">48% of cardiologists treated patients aged 65+.</p>
                    
                    </body>
                    </html>"""]
        ]).toDF("text")

    def runTest(self):
        df = self.data

        document_assembler = DocumentAssembler().setInputCol('text').setOutputCol('document')

        action = "clean"
        patterns = ["<[^>]*>"]
        replacement = " "
        policy = "pretty_all"

        document_normalizer = DocumentNormalizer() \
            .setInputCols("document") \
            .setOutputCol("normalizedDocument") \
            .setAction(action) \
            .setPatterns(patterns) \
            .setReplacement(replacement) \
            .setPolicy(policy) \
            .setLowercase(True)

        sentence_detector = SentenceDetector() \
            .setInputCols(["normalizedDocument"]) \
            .setOutputCol("sentence")

        regex_tokenizer = Tokenizer() \
            .setInputCols(["sentence"]) \
            .setOutputCol("token") \
            .fit(df)

        doc_normalizer_pipeline = \
            Pipeline().setStages([document_assembler, document_normalizer, sentence_detector, regex_tokenizer])

        ds = doc_normalizer_pipeline.fit(df).transform(df)

        ds.select("normalizedDocument").show()

