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
from sparknlp.annotator.similarity.document_similarity_ranker import *
from sparknlp.base import *
from test.util import SparkSessionForTest


@pytest.mark.slow
class DocumentSimilarityRankerTestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkSessionForTest.spark

        self.data = SparkSessionForTest.spark.createDataFrame([
            ["First document, this is my first sentence. This is my second sentence."],
            ["Second document, this is my second sentence. This is my second sentence."],
            ["Third document, climate change is arguably one of the most pressing problems of our time."],
            ["Fourth document, climate change is definitely one of the most pressing problems of our time."],
            ["Fifth document, Florence in Italy, is among the most beautiful cities in Europe."],
            ["Sixth document, Florence in Italy, is a very beautiful city in Europe like Lyon in France."],
            ["Seventh document, the French Riviera is the Mediterranean coastline of the southeast corner of France."],
            ["Eighth document, the warmest place in France is the French Riviera coast in Southern France."]
        ]).toDF("text")

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        sentence_detector = SentenceDetector() \
            .setInputCols(["document"]) \
            .setOutputCol("sentence")
        tokenizer = Tokenizer() \
            .setInputCols(["sentence"]) \
            .setOutputCol("token")

        sentence_embeddings = RoBertaSentenceEmbeddings.pretrained() \
            .setInputCols(["document"]) \
            .setOutputCol("sentence_embeddings")

        document_similarity_ranker = DocumentSimilarityRankerApproach() \
            .setInputCols("sentence_embeddings") \
            .setOutputCol("doc_similarity_rankings") \
            .setSimilarityMethod("brp") \
            .setNumberOfNeighbours(10) \
            .setBucketLength(2.0) \
            .setNumHashTables(3) \
            .setVisibleDistances(True) \
            .setIdentityRanking(True)

        document_similarity_ranker_finisher = DocumentSimilarityRankerFinisher() \
            .setInputCols("doc_similarity_rankings") \
            .setOutputCols(
            "finished_doc_similarity_rankings_id",
            "finished_doc_similarity_rankings_neighbors") \
            .setExtractNearestNeighbor(True)

        pipeline = Pipeline(stages=[
            document_assembler,
            sentence_detector,
            tokenizer,
            sentence_embeddings,
            document_similarity_ranker,
            document_similarity_ranker_finisher
        ])

        model = pipeline.fit(self.data)

        (
            model
            .transform(self.data)
            .select("text",
                    "finished_doc_similarity_rankings_id",
                    "finished_doc_similarity_rankings_neighbors")
            .show(10, False)
        )