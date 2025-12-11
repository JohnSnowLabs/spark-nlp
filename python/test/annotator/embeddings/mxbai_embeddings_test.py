#  Copyright 2017-2022 John Snow Labs
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#

import unittest
import pytest

from sparknlp.annotator import *
from sparknlp.base import *
from pyspark.ml import Pipeline
from test.util import SparkContextForTest


@pytest.mark.local
class MxbaiEmbeddingsTestSpec(unittest.TestCase):

    def setUp(self):
        self.spark = SparkContextForTest.spark
        self.tested_annotator = (
            MxbaiEmbeddings.pretrained()
            .setInputCols(["documents"])
            .setOutputCol("Mxbai")
            .setPoolingStrategy("cls_avg")
        )

    def test_run(self):
        data = self.spark.createDataFrame(
            [["hello world"], ["hello moon"]]
        ).toDF("text")

        document_assembler = (
            DocumentAssembler()
            .setInputCol("text")
            .setOutputCol("documents")
        )

        embeddings_finisher = (
            EmbeddingsFinisher()
            .setInputCols("Mxbai")
            .setOutputCols("embeddings")
        )

        Mxbai = self.tested_annotator

        pipeline = Pipeline().setStages([
            document_assembler,
            Mxbai,
            embeddings_finisher
        ])

        results = pipeline.fit(data).transform(data)
        results.selectExpr("explode(embeddings) as result").show()

    @pytest.mark.slow
    def test_end_to_end_pipeline(self):
        test_data = self.spark.createDataFrame([
            ("This is an example sentence for testing embeddings.",),
            ("Each sentence is converted into a numerical representation.",),
            ("Machine learning models use these embeddings for various tasks.",)
        ]).toDF("text")

        document_assembler = (
            DocumentAssembler()
            .setInputCol("text")
            .setOutputCol("document")
        )

        sentence_detector = (
            SentenceDetectorDLModel
            .pretrained("sentence_detector_dl", "en")
            .setInputCols(["document"])
            .setOutputCol("sentences")
        )

        embeddings = (
            self.tested_annotator
            .setInputCols(["sentences"])
            .setOutputCol("Mxbai")
        )

        pipeline = Pipeline().setStages([
            document_assembler,
            sentence_detector,
            embeddings
        ])

        pipeline_model = pipeline.fit(test_data)
        transformed = pipeline_model.transform(test_data)

        transformed.select("text", "Mxbai.embeddings").show()

        embeddings_data = transformed.select("Mxbai.embeddings").collect()

        for row in embeddings_data:
            embeddings_list = row["embeddings"]
            self.assertGreater(len(embeddings_list), 0,
                               "Embeddings should not be empty")

            first_embedding = embeddings_list[0]
            self.assertGreater(len(first_embedding), 0,
                               "Each embedding should have size greater than 0")
