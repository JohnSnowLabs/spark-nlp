#  Copyright 2017-2024 John Snow Labs
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
from test.util import SparkContextForTest


@pytest.mark.local
class NLLBTransformerTextTranslationTestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkContextForTest.spark

    @pytest.mark.slow
    def test_end_to_end_pipeline(self):
        data = self.spark.createDataFrame([
            [1, """What is the capital of France?"""]]).toDF("id", "text")

        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("documents")

        sentence = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx") \
            .setInputCols("documents") \
            .setOutputCol("sentence")

        marian = MarianTransformer \
            .pretrained() \
            .setInputCols("sentence") \
            .setOutputCol("translation") \
            .setMaxInputLength(512) \
            .setMaxOutputLength(50)

        pipeline = Pipeline().setStages([document_assembler, sentence, marian])
        pipeline.fit(data).transform(data).show()

