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

from pyspark.sql.functions import col
from test.util import SparkSessionForTest, SparkContextForTest

from sparknlp.annotator import *
from sparknlp.base import *


class LightPipelineTextSetUp(unittest.TestCase):

    def setUp(self):
        self.spark = SparkSessionForTest.spark
        self.text = "This is a text input"
        self.textDataSet = self.spark.createDataFrame([[self.text]]).toDF("text")

        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        regex_tok = RegexTokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token")

        pipeline = Pipeline().setStages([document_assembler, regex_tok])
        self.model = pipeline.fit(self.textDataSet)


@pytest.mark.fast
class LightPipelineTextInputTest(LightPipelineTextSetUp, unittest.TestCase):

    def setUp(self):
        super().setUp()

    def runTest(self):
        light_pipeline = LightPipeline(self.model)

        annotations_result = light_pipeline.fullAnnotate(self.text)

        self.assertEqual(len(annotations_result), 1)
        for result in annotations_result:
            self.assertTrue(len(result["document"]) > 0)
            self.assertTrue(len(result["token"]) > 0)

        texts = [self.text, self.text]
        annotations_result = light_pipeline.fullAnnotate(texts)

        self.assertEqual(len(annotations_result), len(texts))
        for result in annotations_result:
            self.assertTrue(len(result["document"]) > 0)
            self.assertTrue(len(result["token"]) > 0)


class LightPipelineImageSetUp(unittest.TestCase):

    def setUp(self):
        self.images_path = os.getcwd() + "/../src/test/resources/image/"
        self.data = SparkSessionForTest.spark.read.format("image") \
            .load(path=self.images_path)

        image_assembler = ImageAssembler() \
            .setInputCol("image") \
            .setOutputCol("image_assembler")

        image_classifier = ViTForImageClassification \
            .pretrained() \
            .setInputCols("image_assembler") \
            .setOutputCol("class")

        pipeline = Pipeline(stages=[
            image_assembler,
            image_classifier,
        ])

        self.vit_model = pipeline.fit(self.data)


@pytest.mark.slow
class LightPipelineImageInputTest(LightPipelineImageSetUp, unittest.TestCase):

    def setUp(self):
        super().setUp()

    def runTest(self):
        image = self.images_path + "hippopotamus.JPEG"
        light_pipeline = LightPipeline(self.vit_model)

        annotations_result = light_pipeline.fullAnnotateImage(image)

        self.assertEqual(len(annotations_result), 1)
        self.assertAnnotations(annotations_result)

        annotations_result = light_pipeline.fullAnnotate(image)

        self.assertEqual(len(annotations_result), 1)
        self.assertAnnotations(annotations_result)

    def assertAnnotations(self, annotations_result):
        for result in annotations_result:
            self.assertTrue(len(result["image_assembler"]) > 0)
            self.assertTrue(len(result["class"]) > 0)


@pytest.mark.slow
class LightPipelineImagesInputTest(LightPipelineImageSetUp, unittest.TestCase):

    def setUp(self):
        super().setUp()

    def runTest(self):
        images = [self.images_path + "hippopotamus.JPEG", self.images_path + "egyptian_cat.jpeg"]
        light_pipeline = LightPipeline(self.vit_model)

        annotations_result = light_pipeline.fullAnnotate(images)

        self.assertEqual(len(annotations_result), len(images))
        self.assertAnnotations(annotations_result)

        annotations_result = light_pipeline.fullAnnotate(images)

        self.assertEqual(len(annotations_result), len(images))
        self.assertAnnotations(annotations_result)

    def assertAnnotations(self, annotations_result):
        for result in annotations_result:
            self.assertTrue(len(result["image_assembler"]) > 0)
            self.assertTrue(len(result["class"]) > 0)


@pytest.mark.slow
class LightPipelineAudioInputTest(unittest.TestCase):

    def setUp(self):
        audio_json = os.getcwd() + "/../src/test/resources/audio/json/audio_floats.json"
        audio_csv = os.getcwd() + "/../src/test/resources/audio/csv/audio_floats.csv"
        self.data = SparkSessionForTest.spark.read.option("inferSchema", value=True).json(audio_json) \
            .select(col("float_array").cast("array<float>").alias("audio_content"))
        self.audio_data = list()
        audio_file = open(audio_csv, 'r')
        csv_lines = audio_file.readlines()
        for csv_line in csv_lines:
            clean_line = float(csv_line.split(',')[0])
            self.audio_data.append(clean_line)

    def runTest(self):
        audio_assembler = AudioAssembler() \
            .setInputCol("audio_content") \
            .setOutputCol("audio_assembler")

        speech_to_text = Wav2Vec2ForCTC \
            .pretrained() \
            .setInputCols("audio_assembler") \
            .setOutputCol("text")

        pipeline = Pipeline(stages=[audio_assembler, speech_to_text])
        self.model = pipeline.fit(self.data)

        light_pipeline = LightPipeline(self.model)

        annotations_result = light_pipeline.fullAnnotate(self.audio_data)

        self.assertEqual(len(annotations_result), 1)
        self.assertAnnotations(annotations_result)

        self.audios = [self.audio_data, self.audio_data]
        annotations_result = light_pipeline.fullAnnotate(self.audios)

        self.assertEqual(len(annotations_result), 2)
        self.assertAnnotations(annotations_result)

    def assertAnnotations(self, annotations_result):
        for result in annotations_result:
            self.assertTrue(len(result["audio_assembler"]) > 0)
            self.assertTrue(len(result["text"]) > 0)


@pytest.mark.slow
class LightPipelineTapasInputTest(unittest.TestCase):

    def setUp(self):
        table_json_source = os.getcwd() + "/../src/test/resources/tapas/rich_people.json"
        with open(table_json_source, "rt") as F:
            self.table = "".join(F.readlines())

        self.question1 = "Who earns 100,000,000?"
        self.question2 = "How much people earn?"
        self.data = SparkContextForTest.spark.createDataFrame([
            [self.table, self.question1],
            [self.table, self.question2]
        ]).toDF("table_json", "questions")

    def runTest(self):
        document_assembler = MultiDocumentAssembler() \
            .setInputCols("table_json", "questions") \
            .setOutputCols("document_questions", "document_table")

        sentence_detector = SentenceDetector() \
            .setInputCols(["document_questions"]) \
            .setOutputCol("questions")

        table_assembler = TableAssembler() \
            .setInputCols(["document_table"]) \
            .setOutputCol("table")

        tapas = TapasForQuestionAnswering() \
            .pretrained() \
            .setMaxSentenceLength(512) \
            .setInputCols(["questions", "table"]) \
            .setOutputCol("answers")

        pipeline = Pipeline(stages=[
            document_assembler,
            sentence_detector,
            table_assembler,
            tapas
        ])

        model = pipeline.fit(self.data)

        light_pipeline = LightPipeline(model)
        annotations_result = light_pipeline.fullAnnotate(self.question1, self.table)

        self.assertEqual(len(annotations_result), 1)
        self.assertAnnotations(annotations_result)

        questions = [self.question1, self.question2]
        annotations_result = light_pipeline.fullAnnotate(questions, [self.table, self.table])

        self.assertEqual(len(annotations_result), len(questions))
        self.assertAnnotations(annotations_result)

    def assertAnnotations(self, annotations_result):
        for result in annotations_result:
            self.assertTrue(len(result["document_questions"]) > 0)
            self.assertTrue(len(result["document_table"]) > 0)
            self.assertTrue(len(result["questions"]) > 0)
            self.assertTrue(len(result["table"]) > 0)
            self.assertTrue(len(result["answers"]) > 0)


@pytest.mark.slow
class LightPipelineQAInputTest(unittest.TestCase):

    def setUp(self):
        self.question = "What's my name?"
        self.context = "My name is Clara and I live in Berkeley."
        self.data = SparkContextForTest.spark.createDataFrame([[self.question, self.context]])\
            .toDF("question", "context")

    def runTest(self):
        document_assembler = MultiDocumentAssembler() \
            .setInputCols(["question", "context"]) \
            .setOutputCols(["document_question", "document_context"])

        qa_classifier = DistilBertForQuestionAnswering.pretrained() \
            .setInputCols(["document_question", 'document_context']) \
            .setOutputCol("answer")

        pipeline = Pipeline().setStages([
            document_assembler,
            qa_classifier
        ])

        model = pipeline.fit(self.data)

        light_pipeline = LightPipeline(model)
        annotations_result = light_pipeline.fullAnnotate(self.question, self.context)

        self.assertEqual(len(annotations_result), 1)
        self.assertAnnotations(annotations_result)

        questions = [self.question, self.question]
        contexts = [self.context, self.context]
        annotations_result = light_pipeline.fullAnnotate(questions, contexts)

        self.assertEqual(len(annotations_result), 2)
        self.assertAnnotations(annotations_result)

    def assertAnnotations(self, annotations_result):
        for result in annotations_result:
            self.assertTrue(len(result["document_question"]) > 0)
            self.assertTrue(len(result["document_context"]) > 0)
            self.assertTrue(len(result["answer"]) > 0)


@pytest.mark.fast
class LightPipelineWrongInputTest(LightPipelineTextSetUp, unittest.TestCase):

    def setUp(self):
        super().setUp()

    def runTest(self):
        light_pipeline = LightPipeline(self.model)

        with self.assertRaises(TypeError):
            light_pipeline.fullAnnotate(1)

        with self.assertRaises(TypeError):
            light_pipeline.fullAnnotate([1, 2])

        with self.assertRaises(TypeError):
            light_pipeline.fullAnnotate({"key": "value"})
