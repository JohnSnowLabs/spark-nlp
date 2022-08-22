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
@pytest.mark.skip(reason="Needs to be Fixed.")
class GraphExtractionTestSpec(unittest.TestCase):

    def setUp(self):
        self.spark = SparkContextForTest.spark
        self.data_set = self.spark.createDataFrame([["Peter Parker is a nice person and lives in New York"]]).toDF(
            "text")

    def runTest(self):
        document_assembler = DocumentAssembler().setInputCol("text").setOutputCol("document")
        tokenizer = Tokenizer().setInputCols("document").setOutputCol("token")

        word_embeddings = WordEmbeddingsModel.pretrained() \
            .setInputCols(["document", "token"]) \
            .setOutputCol("embeddings")

        ner_model = NerDLModel.pretrained() \
            .setInputCols(["document", "token", "embeddings"]) \
            .setOutputCol("ner")

        pos_tagger = PerceptronModel.pretrained() \
            .setInputCols(["document", "token"]) \
            .setOutputCol("pos")

        dependency_parser = DependencyParserModel.pretrained() \
            .setInputCols(["document", "pos", "token"]) \
            .setOutputCol("dependency")

        typed_dependency_parser = TypedDependencyParserModel.pretrained() \
            .setInputCols(["token", "pos", "dependency"]) \
            .setOutputCol("labdep")

        graph_extraction = GraphExtraction() \
            .setInputCols(["document", "token", "ner"]) \
            .setOutputCol("graph") \
            .setRelationshipTypes(["person-PER", "person-LOC"])

        graph_finisher = GraphFinisher() \
            .setInputCol("graph") \
            .setOutputCol("finisher")

        pipeline = Pipeline().setStages([document_assembler, tokenizer,
                                         word_embeddings, ner_model, pos_tagger,
                                         dependency_parser, typed_dependency_parser])

        test_data_set = pipeline.fit(self.data_set).transform(self.data_set)
        pipeline_finisher = Pipeline().setStages([graph_extraction, graph_finisher])

        graph_data_set = pipeline_finisher.fit(test_data_set).transform(test_data_set)
        graph_data_set.show(truncate=False)

