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
from pyspark.sql import SparkSession
from sparknlp.annotator import *
from sparknlp.annotator.similarity.document_similarity_ranker import DocumentSimilarityRankerApproach
from sparknlp.base import *

# from test.util import SparkSessionForTest


@pytest.mark.fast
class DocumentSimilarityRankerTestSpec(unittest.TestCase):

    def setUp(self):
        jars_path = "/Users/stefanolori/workspace/dev/oth/spark-nlp/python/sparknlp/lib/sparknlp.jar"
        spark = SparkSession.builder \
            .master("local[*]") \
            .config("spark.jars", jars_path) \
            .config("spark.driver.memory", "12G") \
            .config("spark.driver.maxResultSize", "2G") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.kryoserializer.buffer.max", "500m") \
            .getOrCreate()

        spark.sparkContext.setLogLevel("WARN")

        # FIXME rollback the setting up from utility class for test
        # self.data = SparkSessionForTest.spark.createDataFrame([
        self.data = spark.createDataFrame([
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

        # TODO add document_similarity_ranker with input col embeddings too
        document_similarity_ranker = DocumentSimilarityRankerApproach() \
            .setInputCols("sentence_embeddings") \
            .setOutputCol("doc_similarity_rankings") \
            .setSimilarityMethod("brp") \
            .setNumberOfNeighbours(10) \
            .setBucketLength(2.0) \
            .setNumHashTables(3) \
            .setVisibleDistances(True) \
            .setIdentityRanking(False)

        pipeline = Pipeline(stages=[
            document_assembler,
            sentence_detector,
            tokenizer,
            sentence_embeddings,
            document_similarity_ranker
            # TODO add document_similarity_ranker_finisher
        ])

        model = pipeline.fit(self.data)
        # TODO add write/read pipeline
        model.transform(self.data).show()

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

        # TODO add document_similarity_ranker with input col embeddings too
        document_similarity_ranker = DocumentSimilarityRankerApproach() \
            .setInputCols("sentence_embeddings") \
            .setOutputCol("doc_similarity_rankings") \
            .setSimilarityMethod("mh") \
            .setNumberOfNeighbours(10) \
            .setNumHashTables(3) \
            .setVisibleDistances(True) \
            .setIdentityRanking(False)

        pipeline = Pipeline(stages=[
            document_assembler,
            sentence_detector,
            tokenizer,
            sentence_embeddings,
            document_similarity_ranker
            # TODO add document_similarity_ranker_finisher
        ])

        model = pipeline.fit(self.data)
        # TODO add write/read pipeline
        transformed = model.transform(self.data)
        transformed.show()

    # FIXME encoding on GloVe generates different embeddings length
    # def runTest(self):
    #     document_assembler = DocumentAssembler() \
    #         .setInputCol("text") \
    #         .setOutputCol("document")
    #     sentence_detector = SentenceDetector() \
    #         .setInputCols(["document"]) \
    #         .setOutputCol("sentence")
    #     tokenizer = Tokenizer() \
    #         .setInputCols(["sentence"]) \
    #         .setOutputCol("token")
    #
    #     glove = WordEmbeddingsModel.pretrained() \
    #         .setInputCols(["sentence", "token"]) \
    #         .setOutputCol("embeddings")
    #
    #     sentence_embeddings = SentenceEmbeddings() \
    #         .setInputCols(["sentence", "embeddings"]) \
    #         .setOutputCol("sentence_embeddings") \
    #         .setPoolingStrategy("AVERAGE")
    #
    #     document_similarity_ranker = DocumentSimilarityRankerApproach() \
    #         .setInputCols("sentence_embeddings") \
    #         .setOutputCol("doc_similarity_rankings") \
    #         .setSimilarityMethod("brp") \
    #         .setNumberOfNeighbours(10) \
    #         .setBucketLength(2.0) \
    #         .setNumHashTables(3) \
    #         .setVisibleDistances(True) \
    #         .setIdentityRanking(True)
    #
    #     print(document_similarity_ranker.__dict__)
    #
    #     # documentSimilarityFinisher = (
    #     #     DocumentSimilarityRankerFinisher()
    #     #         .setInputCols("doc_similarity_rankings")
    #     #         .setOutputCols(
    #     #             "finished_doc_similarity_rankings_id",
    #     #             "finished_doc_similarity_rankings_neighbors")
    #     #         .setExtractNearestNeighbor(True)
    #     # )
    #
    #     pipeline = Pipeline(stages=[
    #         document_assembler,
    #         sentence_detector,
    #         tokenizer,
    #         glove,
    #         sentence_embeddings,
    #         document_similarity_ranker
    #     ])
    #
    #     model = pipeline.fit(self.data)
    #     # model.write().overwrite().save("./tmp_model")
    #     # loaded_model = model.load("./tmp_model")
    #     # loaded_model.transform(self.data).show()
    #     model.transform(self.data).show()
