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
from test.util import SparkContextForTest


@pytest.mark.slow
class UAEEmbeddingsTestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkContextForTest.spark
        self.tested_annotator = UAEEmbeddings \
            .loadSavedModel("/home/ducha/Workspace/JSL/spark-nlp-dev-things/hf_exports/UAE/exported_onnx",
                            SparkContextForTest.spark) \
            .setInputCols(["documents"]) \
            .setOutputCol("embeddings") \
            .setPoolingStrategy("cls_avg")

    def test_run(self):
        data = self.spark.createDataFrame([["hello world"], ["hello moon"]]).toDF("text")

        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("documents")

        embeddings_finisher = EmbeddingsFinisher().setInputCols("embeddings").setOutputCols("embeddings")

        uae = self.tested_annotator

        pipeline = Pipeline().setStages([document_assembler, uae, embeddings_finisher])
        results = pipeline.fit(data).transform(data)

        results.selectExpr("explode(embeddings) as result").show(truncate=False)
