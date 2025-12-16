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
from sparknlp.training import CoNLL
from test.util import SparkSessionForTest


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
        with pytest.raises(Exception) as exc_info:
            pipeline.fit(self.dataset)
        assert "Could not find a suitable tensorflow graph" in str(exc_info.value)

    def test_serializable_in_pipeline(self):
        embeddings, ner_graph_checker, _ = setup_annotators(self.dataset)

        pipeline = Pipeline(stages=[ner_graph_checker, embeddings])
        pipeline.write().overwrite().save("tmp_nerdlgraphchecker_pipeline")
        loaded_pipeline = Pipeline.load("tmp_nerdlgraphchecker_pipeline")

        pipeline_model = loaded_pipeline.fit(self.dataset)
        ner_graph_checker_model = pipeline_model.stages[0]
        ngc_model_params = ner_graph_checker_model.params

        pipeline_model.write().overwrite().save("tmp_nerdlgraphchecker_pipeline_model")

        loaded_pipeline_model = PipelineModel.load(
            "tmp_nerdlgraphchecker_pipeline_model"
        )

        ngc_loaded = loaded_pipeline_model.stages[0]
        ngc_loaded_params = ngc_loaded.params

        assert (
            ngc_model_params == ngc_loaded_params
        ), "Parameters do not match after serialization."

        loaded_pipeline_model.transform(self.dataset).show()

    def test_determine_suitable_graph_before_training(self):
        embeddings_invalid, ner_graph_checker, ner = setup_annotators(
            self.dataset, embeddingDim=101
        )
        pipeline = Pipeline(stages=[embeddings_invalid, ner_graph_checker, ner])
        with pytest.raises(Exception) as exc_info:
            pipeline.fit(self.dataset)
        assert "Could not find a suitable tensorflow graph" in str(exc_info.value)

    def test_fill_column_metadata_with_extracted_params(self):
        """Test that column metadata is filled with extracted params"""

        # Reference implementation with spark
        def get_expected_params(dataset: DataFrame, tokenCol: str, labelsCol: str):
            # extract distinct labels
            labels = (
                dataset.selectExpr(f"explode({labelsCol}.result)").distinct().collect()
            )

            # extract distinct characters
            chars = (
                dataset.selectExpr(f"explode({tokenCol}.result)")
                .rdd.flatMap(lambda tokens: [c for tok in tokens for c in tok])
                .distinct()
                .collect()
            )

            dsLen = dataset.count()

            return set(r[0] for r in labels), set(r[0] for r in chars), dsLen

        embeddings, ner_dl_graph_checker, _ = setup_annotators(self.dataset)
        pipeline = Pipeline(stages=[embeddings, ner_dl_graph_checker])
        fitted = pipeline.fit(self.dataset)
        result = fitted.transform(self.dataset)

        label_col = ner_dl_graph_checker.getLabelColumn()
        label_field = result.schema[label_col]

        assert (
            NerDLGraphCheckerModel.graphParamsMetadataKey in label_field.metadata
        ), "Label column metadata should contain graph params."

        graph_params_meta = label_field.metadata[
            NerDLGraphCheckerModel.graphParamsMetadataKey
        ]

        embeddings_dim = graph_params_meta[NerDLGraphCheckerModel.embeddingsDimKey]
        labels = set(graph_params_meta[NerDLGraphCheckerModel.labelsKey])
        chars = set("".join(graph_params_meta[NerDLGraphCheckerModel.charsKey]))
        ds_len = graph_params_meta[NerDLGraphCheckerModel.dsLenKey]

        expected_embedding_dim = embeddings.getDimension()
        (
            expected_labels,
            expected_chars,
            expected_dsLen,
        ) = get_expected_params(result, "token", "label")

        assert (
            embeddings_dim == expected_embedding_dim
        ), "Extracted embeddings dim should match the embeddings model dimension."
        assert (
            labels == expected_labels
        ), "Extracted labels should match the dataset labels."
        assert (
            chars == expected_chars
        ), "Extracted chars should match the dataset chars."
        assert (
            ds_len == expected_dsLen
        ), "Extracted dataset length should match the dataset length."


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
        with pytest.raises(Exception) as exc_info:
            pipeline.fit(self.dataset)
        assert "Could not find a suitable tensorflow graph" in str(exc_info.value)
