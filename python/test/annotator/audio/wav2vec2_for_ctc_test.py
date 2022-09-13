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
import os
import unittest

import pytest

from sparknlp.annotator import *
from sparknlp.base import *
from pyspark.sql.functions import col

from test.util import SparkSessionForTest


@pytest.mark.fast
class Wav2Vec2ForCTCTestSpec(unittest.TestCase):
    def setUp(self):
        audio_path = os.getcwd() + "/../src/test/resources/audio/json/audio_floats.json"
        self.data = SparkSessionForTest.spark.read.option("inferSchema", value=True).json(audio_path) \
            .select(col("float_array").cast("array<float>").alias("audio_content"))

    def runTest(self):
        self.data.show()
        audio_assembler = AudioAssembler() \
            .setInputCol("audio_content") \
            .setOutputCol("audio_assembler")

        speech_to_text = Wav2Vec2ForCTC \
            .pretrained()\
            .setInputCols("audio_assembler") \
            .setOutputCol("text")

        pipeline = Pipeline(stages=[
            audio_assembler,
            speech_to_text,
        ])

        model = pipeline.fit(self.data)
        result_df = model.transform(self.data)
        assert result_df.select("text").count() > 0
