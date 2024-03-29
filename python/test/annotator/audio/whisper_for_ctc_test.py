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

import os
import pytest
from pyspark.sql.functions import col

from sparknlp.annotator import *
from sparknlp.base import *
from test.util import SparkSessionForTest


@pytest.mark.slow
class WhisperTestSetUp(unittest.TestCase):
    def setUp(self):
        audio_path = os.getcwd() + "/../src/test/resources/audio/json/audio_floats.json"
        self.data = SparkSessionForTest.spark.read.option("inferSchema", value=True).json(audio_path) \
            .select(col("float_array").cast("array<float>").alias("audio_content"))

        self.audio_assembler = AudioAssembler() \
            .setInputCol("audio_content") \
            .setOutputCol("audio_assembler")

        # Edit Manually
        model_path = "exported_onnx/openai/whisper-tiny"
        self.speech_to_text = WhisperForCTC \
            .loadSavedModel(model_path, SparkSessionForTest.spark) \
            .setInputCols("audio_assembler") \
            .setOutputCol("text")

        self.pipeline = Pipeline(stages=[self.audio_assembler, self.speech_to_text])
        self.model = self.pipeline.fit(self.data)
        self.model.write().overwrite().save("./tmp_Whisper_pipeline_model")


@pytest.mark.slow
class WhisperForCTCTestSpec(WhisperTestSetUp, unittest.TestCase):

    def setUp(self):
        super().setUp()

    def runTest(self):
        self.data.show()

        result_df = self.model.transform(self.data)
        result_df.select("text.result").show(truncate=False)
        assert result_df.select("text").count() > 0


@pytest.mark.slow
class LightWhisperForCTCOneAudioTestSpec(WhisperTestSetUp, unittest.TestCase):
    def setUp(self):
        super().setUp()
        audio_path = os.getcwd() + "/../src/test/resources/audio/csv/audio_floats.csv"
        self.audio_data = list()
        audio_file = open(audio_path, 'r')
        csv_lines = audio_file.readlines()
        for csv_line in csv_lines:
            clean_line = float(csv_line.split(',')[0])
            self.audio_data.append(clean_line)

    def runTest(self):
        light_pipeline = LightPipeline(self.model)

        annotations_result = light_pipeline.fullAnnotate(self.audio_data)

        self.assertEqual(len(annotations_result), 1)
        for result in annotations_result:
            self.assertTrue(len(result["audio_assembler"]) > 0)
            self.assertTrue(len(result["text"]) > 0)


@pytest.mark.slow
class LightWhisperForCTCTestSpec(WhisperTestSetUp, unittest.TestCase):
    def setUp(self):
        super().setUp()
        audio_path = os.getcwd() + "/../src/test/resources/audio/csv/audio_floats.csv"
        self.audio_data = list()
        audio_file = open(audio_path, 'r')
        csv_lines = audio_file.readlines()
        for csv_line in csv_lines:
            clean_line = float(csv_line.split(',')[0])
            self.audio_data.append(clean_line)

    def runTest(self):
        light_pipeline = LightPipeline(self.model)
        self.audios = [self.audio_data, self.audio_data]

        annotations_result = light_pipeline.fullAnnotate(self.audios)

        self.assertEqual(len(annotations_result), 2)
        for result in annotations_result:
            self.assertTrue(len(result["audio_assembler"]) > 0)
            self.assertTrue(len(result["text"]) > 0)


@pytest.mark.slow
class WhisperForCTCLangTaskTestSpec(WhisperTestSetUp, unittest.TestCase):
    def setUp(self):
        super().setUp()
        audio_path = os.getcwd() + "/../src/test/resources/audio/txt/librispeech_asr_0.txt"
        with open(audio_path) as file:
            raw_floats = [float(data) for data in file.read().strip().split("\n")]

        self.data = SparkSessionForTest.spark.createDataFrame([[raw_floats]]).toDF("audio_content")

    def runTest(self):
        self.speech_to_text.setLanguage("<|de|>").setTask("<|translate|>")
        pipeline = Pipeline(stages=[self.audio_assembler, self.speech_to_text])

        result = pipeline.fit(self.data).transform(self.data)

        result_text = result.select("text.result").collect()[0]["result"][0]

        expected_text = " Mr. Kfilter is the apostle of the middle classes and we are glad to welcome his gospel."

        assert result_text == expected_text
