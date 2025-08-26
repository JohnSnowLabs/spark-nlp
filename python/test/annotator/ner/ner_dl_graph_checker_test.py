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
from pyspark.ml import Pipeline
import sparknlp
from sparknlp.training import CoNLL
from test.util import SparkSessionForTest
from pyspark.errors.exceptions.captured import IllegalArgumentException


def setup_annotators(dataset, embeddingDim: int = 100):
    # Get GloVe embeddings
    embeddings = (
        WordEmbeddings()
        .setStoragePath(
            "../src/test/resources/ner-corpus/embeddings.100d.test.txt", "TEXT"
        )
        .setDimension(embeddingDim)
        .setInputCols("sentence", "token")
        .setOutputCol("embeddings")
        .setStorageRef(f"embeddings_ner_{embeddingDim}")
        .fit(dataset)
    )

    ner_graph_checker = (
        NerDLGraphChecker()
        .setInputCols(["sentence", "token"])
        .setLabelColumn("label")
        .setEmbeddingsModel(embeddings)
    )

    # NerDLApproach setup
    graph_folder = "../src/test/resources/graph"
    ner = (
        NerDLApproach()
        .setInputCols(["document", "token", "embeddings"])
        .setOutputCol("ner")
        .setLabelColumn("label")
        .setLr(1e-1)
        .setPo(5e-3)
        .setDropout(5e-1)
        .setMaxEpochs(1)
        .setRandomSeed(0)
        .setVerbose(0)
        .setEvaluationLogExtended(True)
        .setEnableOutputLogs(True)
        .setGraphFolder(graph_folder)
        .setUseBestModel(True)
    )

    return embeddings, ner_graph_checker, ner


@pytest.mark.fast
class NerDLGraphCheckerTest(unittest.TestCase):
    def setUp(self) -> None:
        data_path = "../src/test/resources/ner-corpus/test_ner_dataset.txt"

        # Read CoNLL dataset
        self.dataset = CoNLL().readDataset(SparkSessionForTest.spark, data_path)

    def test_find_right_graph(self):
        embeddings, ner_graph_checker, _ = setup_annotators(self.dataset)
        pipeline = Pipeline(stages=[embeddings, ner_graph_checker])
        # Should fit without error if graph matches
        pipeline.fit(self.dataset)

    def test_throw_exception_if_graph_not_found(self):
        embeddings_invalid, ner_graph_checker, _ = setup_annotators(
            self.dataset, embeddingDim=101
        )
        pipeline = Pipeline(stages=[embeddings_invalid, ner_graph_checker])
        with pytest.raises(IllegalArgumentException) as exc_info:
            pipeline.fit(self.dataset)
        assert "Could not find a suitable tensorflow graph" in str(exc_info.value)

    def test_serializable_in_pipeline(self):
        embeddings, ner_graph_checker, _ = setup_annotators(self.dataset)

        pipeline = Pipeline(stages=[ner_graph_checker, embeddings])
        pipeline.write().overwrite().save("tmp_nerdlgraphchecker_pipeline")
        loaded_pipeline = Pipeline.load("tmp_nerdlgraphchecker_pipeline")

        pipeline_model = loaded_pipeline.fit(self.dataset)
        pipeline_model.write().overwrite().save("tmp_nerdlgraphchecker_pipeline_model")

        loaded_pipeline_model = PipelineModel.load(
            "tmp_nerdlgraphchecker_pipeline_model"
        )
        loaded_pipeline_model.transform(self.dataset).show()

    def test_determine_suitable_graph_before_training(self):
        embeddings_invalid, ner_graph_checker, ner = setup_annotators(
            self.dataset, embeddingDim=101
        )
        pipeline = Pipeline(stages=[embeddings_invalid, ner_graph_checker, ner])
        with pytest.raises(IllegalArgumentException) as exc_info:
            pipeline.fit(self.dataset)
        assert "Could not find a suitable tensorflow graph" in str(exc_info.value)


@pytest.mark.slow
class NerDLGraphCheckerBatchAnnotateTest(unittest.TestCase):
    def setUp(self) -> None:
        data_path = "../src/test/resources/ner-corpus/test_ner_dataset.txt"

        # Read CoNLL dataset
        self.dataset = CoNLL().readDataset(SparkSessionForTest.spark, data_path)

    def test_determine_graph_size_with_batch_annotate_annotators(self):
        from sparknlp.annotator import DistilBertEmbeddings

        _, ner_graph_checker, ner = setup_annotators(self.dataset)
        # Use pretrained DistilBert and intentionally set wrong dimension
        distilbert = (
            DistilBertEmbeddings.pretrained("distilbert_base_cased", "en")
            .setInputCols(["sentence", "token"])
            .setOutputCol("embeddings")
        )
        distilbert.setDimension(distilbert.getDimension() + 3)  # Wrong dimension
        pipeline = Pipeline(stages=[ner_graph_checker, distilbert, ner])
        with pytest.raises(IllegalArgumentException) as exc_info:
            pipeline.fit(self.dataset)
        assert "Could not find a suitable tensorflow graph" in str(exc_info.value)
