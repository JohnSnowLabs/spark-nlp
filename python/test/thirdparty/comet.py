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
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.logging.comet import CometLogger
import comet_ml
import os
import glob
from test.util import SparkContextForTest


class CometTestSpec(unittest.TestCase):
    logger = None
    trainDataset = None
    pipeline = None
    classifier = None
    OUTPUT_LOG_PATH = "./comet"
    pipelineModel = None
    viz_path = "../src/test/resources/logging/comet/spark_nlp_display_viz.html"

    @classmethod
    def setUpClass(cls):
        from sparknlp.training import POS

        cls.trainDataset = POS().readDataset(
            SparkContextForTest.spark,
            "../src/test/resources/anc-pos-corpus-small/test-training.txt",
            delimiter="|",
            outputPosCol="tags",
            outputDocumentCol="document",
            outputTextCol="text",
        )

        document_assembler = (
            DocumentAssembler().setInputCol("text").setOutputCol("document")
        )

        tokenizer = Tokenizer().setInputCols("document").setOutputCol("token")

        embds = (
            WordEmbeddings()
            .setStoragePath(
                "../src/test/resources/ner-corpus/embeddings.100d.test.txt", ReadAs.TEXT
            )
            .setDimension(100)
            .setStorageRef("glove_100d")
            .setInputCols("document", "token")
            .setOutputCol("embeddings")
        )

        cls.classifier = (
            NerDLApproach()
            .setInputCols("document", "token", "embeddings")
            .setLabelColumn("tags")
            .setOutputCol("out")
            .setMaxEpochs(1)
            .setEnableOutputLogs(True)
            .setOutputLogsPath(cls.OUTPUT_LOG_PATH)
        )

        cls.pipeline = Pipeline(
            stages=[document_assembler, tokenizer, embds, cls.classifier]
        )

        # TODO: Does not work for SentenceDetectorDLApproach due to log naming scheme.
        # cls.trainDataset = SparkContextForTest.spark.createDataFrame(
        #     [["This is a sentence."]], ["text"]
        # )
        #
        # document_assembler = (
        #     DocumentAssembler().setInputCol("text").setOutputCol("document")
        # )
        # cls.classifier = (
        #     SentenceDetectorDLApproach()
        #     .setInputCols("document")
        #     .setOutputCol("sentence")
        #     .setOutputLogsPath(cls.OUTPUT_LOG_PATH)
        #     .setEpochsNumber(1)
        # )
        #
        # cls.pipeline = Pipeline(stages=[document_assembler, cls.classifier])

        comet_ml.init(project_name="sparknlp-testing", offline_directory="/tmp")
        cls.logger = CometLogger(
            comet_mode="offline", offline_directory=cls.OUTPUT_LOG_PATH
        )

    @classmethod
    def tearDownClass(cls):
        cls.logger.end()

    def test_monitor(self):
        self.logger.monitor(self.OUTPUT_LOG_PATH, self.classifier, 1)
        self.pipelineModel = self.pipeline.fit(self.trainDataset)

        test_data = SparkContextForTest.spark.createDataFrame([["Hello!"]], ["text"])
        self.pipelineModel.transform(test_data).select("out").show(truncate=False)

    def test_log_pipeline_parameters(self):
        if self.pipelineModel:
            self.logger.log_pipeline_parameters(self.pipelineModel)
        else:
            raise ValueError("Pipeline was not trained in test_monitor.")

    def test_eval_logging(self):
        metrics = {
            "label1": {
                "precision": 0.5,
                "recall": 1.0,
                "f1-score": 0.67,
                "support": 1,
            },
            "label2": {
                "precision": 0.0,
                "recall": 0.0,
                "f1-score": 0.0,
                "support": 1,
            },
        }
        for key, value in metrics.items():
            self.logger.log_metrics(value, prefix=key)

    def test_visualization_logging(self):
        with open(self.viz_path) as viz:
            viz = viz.read()
        self.logger.log_visualization(viz, name=f"spark_nlp_display_viz.html")

    def test_completed_run(self):
        list_of_files = glob.glob(f"{self.OUTPUT_LOG_PATH}/*.log")
        latest_log = max(list_of_files, key=os.path.getctime)
        self.logger.log_completed_run(latest_log)

    def test_log_asset(self):
        self.logger.log_asset(self.viz_path)

    def test_experiment_id(self):
        self.logger.end()
        self.logger = CometLogger(
            comet_mode="offline",
            offline_directory=self.OUTPUT_LOG_PATH,
            experiment_id="7d8605d837d84489af9f2f742f69b39c",
        )

    def runTest(self):
        self.test_monitor()
        self.test_log_pipeline_parameters()
        self.test_eval_logging()
        self.test_visualization_logging()
        self.test_completed_run()
        self.test_log_asset()
        self.test_experiment_id()
