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
from test.util import SparkSessionForTest


@pytest.mark.fast
class RegexTokenizerTestSpec(unittest.TestCase):
    def setUp(self) -> None:
        self.data = SparkSessionForTest.spark.createDataFrame(
            [["AL 123456!, TX 54321-4444, AL :55555-4444, 12345-4444, 12345"]]
        ).toDF("text")

    def runTest(self) -> None:
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        sentence_detector = SentenceDetector() \
            .setInputCols(["document"]) \
            .setOutputCol("sentence")

        pattern = "^(\\s+)|(?=[\\s+\"\'\|:;<=>!?~{}*+,$)\(&%\\[\\]])|(?<=[\\s+\"\'\|:;<=>!?~{}*+,$)\(&%\\[\\]])|(?=\.$)"

        regex_tok = RegexTokenizer() \
            .setInputCols(["sentence"]) \
            .setOutputCol("regex_token") \
            .setPattern(pattern) \
            .setTrimWhitespace(False) \
            .setPreservePosition(True)

        pipeline = Pipeline().setStages([document_assembler, sentence_detector, regex_tok])

        pipeline_model = pipeline.fit(self.data)

        pipe_path = "file:///" + os.getcwd() + "/tmp_regextok_pipeline"
        pipeline_model.write().overwrite().save(pipe_path)

        loaded_pipeline: PipelineModel = PipelineModel.read().load(pipe_path)
        loaded_pipeline.transform(self.data).show()

