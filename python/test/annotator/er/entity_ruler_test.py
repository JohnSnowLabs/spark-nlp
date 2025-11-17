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
from test.util import SparkContextForTest


@pytest.mark.fast
class EntityRulerTestSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.spark.createDataFrame([["John Snow lives in Winterfell"]]).toDF("text")
        self.path = os.getcwd() + "/../src/test/resources/entity-ruler/keywords_only.json"

    def runTest(self):
        document_assembler = DocumentAssembler().setInputCol("text").setOutputCol("document")
        tokenizer = Tokenizer().setInputCols("document").setOutputCol("token")

        entity_ruler = EntityRulerApproach() \
            .setInputCols(["document", "token"]) \
            .setOutputCol("entity") \
            .setPatternsResource(self.path)

        pipeline = Pipeline(stages=[document_assembler, tokenizer, entity_ruler])
        model = pipeline.fit(self.data)
        result = model.transform(self.data)
        self.assertTrue(result.select("entity").count() > 0)


@pytest.mark.fast
class EntityRulerOneColumnTestSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.spark.createDataFrame([["John Snow lives in Winterfell"]]).toDF("text")
        self.path = os.getcwd() + "/../src/test/resources/entity-ruler/keywords_only.json"

    def runTest(self):
        document_assembler = DocumentAssembler().setInputCol("text").setOutputCol("document")

        entity_ruler = EntityRulerApproach() \
            .setInputCols("document") \
            .setOutputCol("entity") \
            .setPatternsResource(self.path)

        pipeline = Pipeline(stages=[document_assembler, entity_ruler])
        model = pipeline.fit(self.data)
        result = model.transform(self.data)
        self.assertTrue(result.select("entity").count() > 0)


@pytest.mark.fast
class EntityRulerLightPipelineTestSpec(unittest.TestCase):
    def setUp(self):
        self.empty_df = SparkContextForTest.spark.createDataFrame([[""]]).toDF("text")
        self.path = os.getcwd() + "/../src/test/resources/entity-ruler/url_regex.json"

    def runTest(self):
        document_assembler = DocumentAssembler().setInputCol("text").setOutputCol("document")
        tokenizer = Tokenizer().setInputCols('document').setOutputCol('token')

        entity_ruler = EntityRulerApproach() \
            .setInputCols(["document", "token"]) \
            .setOutputCol("entity") \
            .setPatternsResource(self.path)

        pipeline = Pipeline(stages=[document_assembler, tokenizer, entity_ruler])
        pipeline_model = pipeline.fit(self.empty_df)
        light_pipeline = LightPipeline(pipeline_model)
        result = light_pipeline.annotate("This is Google's URI http://google.com. And this is Yahoo's URI http://yahoo.com")

        self.assertTrue(len(result["entity"]) == 2)

@pytest.mark.fast
class EntityRulerAutoModeTestSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.spark.createDataFrame(
            [["This server list includes 192.168.1.1, 10.0.0.45 and 172.16.0.2 for internal routing."]]
        ).toDF("text")

    def runTest(self):
        document_assembler = DocumentAssembler().setInputCol("text").setOutputCol("document")
        tokenizer = Tokenizer().setInputCols("document").setOutputCol("token")

        entity_ruler = EntityRulerModel() \
            .setInputCols(["document", "token"]) \
            .setOutputCol("entity") \
            .setAutoMode("NETWORK_ENTITIES")

        pipeline = Pipeline(stages=[document_assembler, tokenizer, entity_ruler])
        pipeline_model = pipeline.fit(self.data)
        result = pipeline_model.transform(self.data)

        self.assertTrue(result.select("entity").count() > 0)
