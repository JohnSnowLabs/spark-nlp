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

from sparknlp.base import *
from sparknlp.annotator import *
from test.util import SparkSessionForTest


@pytest.mark.fast
class MultiDocumentAssemblerTest(unittest.TestCase):

    def setUp(self):
        self.spark = SparkSessionForTest.spark
        self.input1 = "This is the first input"
        self.input2 = "This is the second input"
        self.twoInputDataSet = self.spark.createDataFrame([[self.input1, self.input2]]) \
            .toDF("input1", "input2")

    def runTest(self):

        multi_document_assembler = MultiDocumentAssembler() \
            .setInputCols("input1", "input2") \
            .setOutputCols("output1", "output2")

        tokenizer = Tokenizer() \
            .setInputCols("output2") \
            .setOutputCol("token")

        pipeline = Pipeline().setStages([multi_document_assembler, tokenizer])
        result_df = pipeline.fit(self.twoInputDataSet).transform(self.twoInputDataSet)

        output1 = result_df.select("output1.result").collect()[0]
        output2 = result_df.select("output2.result").collect()[0]
        tokens = result_df.select("token.result").collect()[0]

        assert len(output1) > 0
        assert len(output2) > 0
        assert len(tokens) > 0


@pytest.mark.fast
class LightFullAnnotateMultiDocumentAssemblerTest(unittest.TestCase):

    def setUp(self):
        self.spark = SparkSessionForTest.spark
        self.input1 = "This is the first input"
        self.input2 = "This is the second input"
        self.twoInputDataSet = self.spark.createDataFrame([[self.input1, self.input2]]) \
            .toDF("input1", "input2")

    def runTest(self):

        multi_document_assembler = MultiDocumentAssembler() \
            .setInputCols("input1", "input2") \
            .setOutputCols("output1", "output2")

        tokenizer = Tokenizer() \
            .setInputCols("output2") \
            .setOutputCol("token")

        pipeline = Pipeline().setStages([multi_document_assembler, tokenizer])
        model = pipeline.fit(self.twoInputDataSet)
        light_pipeline = LightPipeline(model)

        actual_result = light_pipeline.fullAnnotate(self.input1, self.input2)
        expected_result = [{
            "output1": [Annotation("document", 0, len(self.input1) - 1, self.input1, {}, [])],
            "output2": [Annotation("document", 0, len(self.input2) - 1, self.input2, {}, [])],
            "token": [
                Annotation("token", 0, 3, "This", {"sentence": "0"}, []),
                Annotation("token", 5, 6, "is", {"sentence": "0"}, []),
                Annotation("token", 8, 10, "the", {"sentence": "0"}, []),
                Annotation("token", 12, 17, "second", {"sentence": "0"}, []),
                Annotation("token", 19, 23, "input", {"sentence": "0"}, [])
            ]
        }]

        assert actual_result == expected_result

        actual_result2 = light_pipeline.fullAnnotate([[self.input1, self.input2], [self.input1, self.input2]])
        expected_result2 = [{
            "output1": [Annotation("document", 0, len(self.input1) - 1, self.input1, {}, [])],
            "output2": [Annotation("document", 0, len(self.input2) - 1, self.input2, {}, [])],
            "token": [
                Annotation("token", 0, 3, "This", {"sentence": "0"}, []),
                Annotation("token", 5, 6, "is", {"sentence": "0"}, []),
                Annotation("token", 8, 10, "the", {"sentence": "0"}, []),
                Annotation("token", 12, 17, "second", {"sentence": "0"}, []),
                Annotation("token", 19, 23, "input", {"sentence": "0"}, [])
            ]
            },
            {
                "output1": [Annotation("document", 0, len(self.input1) - 1, self.input1, {}, [])],
                "output2": [Annotation("document", 0, len(self.input2) - 1, self.input2, {}, [])],
                "token": [
                    Annotation("token", 0, 3, "This", {"sentence": "0"}, []),
                    Annotation("token", 5, 6, "is", {"sentence": "0"}, []),
                    Annotation("token", 8, 10, "the", {"sentence": "0"}, []),
                    Annotation("token", 12, 17, "second", {"sentence": "0"}, []),
                    Annotation("token", 19, 23, "input", {"sentence": "0"}, [])
                ]
            }
        ]

        assert actual_result2 == expected_result2


@pytest.mark.fast
class LightAnnotateMultiDocumentAssemblerTest(unittest.TestCase):

    def setUp(self):
        self.spark = SparkSessionForTest.spark
        self.input1 = "This is the first input"
        self.input2 = "This is the second input"
        self.twoInputDataSet = self.spark.createDataFrame([[self.input1, self.input2]]) \
            .toDF("input1", "input2")

    def runTest(self):

        multi_document_assembler = MultiDocumentAssembler() \
            .setInputCols("input1", "input2") \
            .setOutputCols("output1", "output2")

        tokenizer = Tokenizer() \
            .setInputCols("output2") \
            .setOutputCol("token")

        pipeline = Pipeline().setStages([multi_document_assembler, tokenizer])
        model = pipeline.fit(self.twoInputDataSet)
        light_pipeline = LightPipeline(model)

        actual_result = light_pipeline.annotate(self.input1, self.input2)
        expected_result = {
            "output1": [self.input1],
            "output2": [self.input2],
            "token": ["This", "is", "the", "second", "input"]
        }

        assert actual_result == expected_result

        actual_result2 = light_pipeline.annotate([[self.input1, self.input2], [self.input1, self.input2]])
        expected_result2 = [{
            "output1": [self.input1],
            "output2": [self.input2],
            "token": ["This", "is", "the", "second", "input"]
            },
            {
            "output1": [self.input1],
            "output2": [self.input2],
            "token": ["This", "is", "the", "second", "input"]
            }
        ]

        assert actual_result2 == expected_result2
