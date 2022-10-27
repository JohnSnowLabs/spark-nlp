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

from sparknlp.pretrained import PretrainedPipeline


@pytest.mark.slow
class PretrainedPipelineTextInputTest(unittest.TestCase):

    def setUp(self):
        self.pipeline = PretrainedPipeline("clean_slang", "en")

    def runTest(self):
        slang_text = "yo, what is wrong with ya?"
        annotations_result = self.pipeline.fullAnnotate(slang_text)

        self.assertEqual(len(annotations_result), 1)
        for result in annotations_result:
            self.assertTrue(len(result) > 0)

        slang_texts = [slang_text, slang_text]
        annotations_result = self.pipeline.fullAnnotate(slang_texts)
        self.assertEqual(len(annotations_result), len(slang_texts))
        for result in annotations_result:
            self.assertTrue(len(result) > 0)


class PretrainedPipelineImageSetUp(unittest.TestCase):

    def setUp(self):
        self.images_path = os.getcwd() + "/../src/test/resources/image/"
        self.pipeline = PretrainedPipeline("pipeline_image_classifier_vit_dogs", "en")


@pytest.mark.slow
class PretrainedPipelineImageInputTest(PretrainedPipelineImageSetUp, unittest.TestCase):

    def setUp(self):
        super(PretrainedPipelineImageInputTest, self).setUp()

    def runTest(self):
        image = self.images_path + "chihuahua.jpg"

        annotations_result = self.pipeline.fullAnnotate(image)

        self.assertEqual(len(annotations_result), 1)
        for result in annotations_result:
            self.assertTrue(len(result) > 0)

        images = [self.images_path + "chihuahua.jpg", self.images_path + "egyptian_cat.jpeg"]

        annotations_result = self.pipeline.fullAnnotate(images)

        self.assertEqual(len(annotations_result), len(images))
        for result in annotations_result:
            self.assertTrue(len(result) > 0)


@pytest.mark.slow
class PretrainedPipelineImagesInputTest(PretrainedPipelineImageSetUp, unittest.TestCase):

    def setUp(self):
        super(PretrainedPipelineImagesInputTest, self).setUp()

    def runTest(self):
        image = self.images_path + "chihuahua.jpg"

        annotations_result = self.pipeline.fullAnnotateImage(image)

        self.assertEqual(len(annotations_result), 1)
        for result in annotations_result:
            self.assertTrue(len(result) > 0)

        images = [self.images_path + "chihuahua.jpg", self.images_path + "egyptian_cat.jpeg"]

        annotations_result = self.pipeline.fullAnnotateImage(images)

        self.assertEqual(len(annotations_result), len(images))
        for result in annotations_result:
            self.assertTrue(len(result) > 0)


@pytest.mark.slow
class PretrainedPipelineAudioInputTest(unittest.TestCase):

    def setUp(self):
        audio_csv = os.getcwd() + "/../src/test/resources/audio/csv/audio_floats.csv"
        self.audio_data = list()
        audio_file = open(audio_csv, 'r')
        csv_lines = audio_file.readlines()
        for csv_line in csv_lines:
            clean_line = float(csv_line.split(',')[0])
            self.audio_data.append(clean_line)

        self.pipeline = PretrainedPipeline('pipeline_asr_wav2vec2_bilal_2022', lang='en')

    def runTest(self):
        annotations_result = self.pipeline.fullAnnotate(self.audio_data)

        self.assertEqual(len(annotations_result), 1)
        for result in annotations_result:
            self.assertTrue(len(result) > 0)

        audios = [self.audio_data, self.audio_data]
        annotations_result = self.pipeline.fullAnnotate(audios)

        self.assertEqual(len(annotations_result), len(audios))
        for result in annotations_result:
            self.assertTrue(len(result) > 0)

