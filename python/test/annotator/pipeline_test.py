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


@pytest.mark.slow
@pytest.mark.skip(reason="Needs to be Fixed.")
class PipelineTestSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.data

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token")
        lemmatizer = Lemmatizer() \
            .setInputCols(["token"]) \
            .setOutputCol("lemma") \
            .setDictionary("file:///" + os.getcwd() + "/../src/test/resources/lemma-corpus-small/simple.txt",
                           key_delimiter="->", value_delimiter="\t")
        finisher = Finisher() \
            .setInputCols(["token", "lemma"]) \
            .setOutputCols(["token_views", "lemma_views"]) \
            .setOutputAsArray(False) \
            .setAnnotationSplitSymbol('@') \
            .setValueSplitSymbol('#')
        pipeline = Pipeline(stages=[document_assembler, tokenizer, lemmatizer, finisher])
        model = pipeline.fit(self.data)
        token_before_save = model.transform(self.data).select("token_views").take(1)[0].token_views.split("@")[2]
        lemma_before_save = model.transform(self.data).select("lemma_views").take(1)[0].lemma_views.split("@")[2]
        pipe_path = "file:///" + os.getcwd() + "/tmp_pipeline"
        pipeline.write().overwrite().save(pipe_path)
        loaded_pipeline = Pipeline.read().load(pipe_path)
        token_after_save = model.transform(self.data).select("token_views").take(1)[0].token_views.split("@")[2]
        lemma_after_save = model.transform(self.data).select("lemma_views").take(1)[0].lemma_views.split("@")[2]
        assert token_before_save == "sad"
        assert lemma_before_save == "unsad"
        assert token_after_save == token_before_save
        assert lemma_after_save == lemma_before_save
        pipeline_model = loaded_pipeline.fit(self.data)
        pipeline_model.transform(self.data).show()
        pipeline_model.write().overwrite().save(pipe_path)
        loaded_model = PipelineModel.read().load(pipe_path)
        loaded_model.transform(self.data).show()
        locdata = list(map(lambda d: d[0], self.data.select("text").collect()))
        spless = LightPipeline(loaded_model).annotate(locdata)
        fullSpless = LightPipeline(loaded_model).fullAnnotate(locdata)
        for row in spless[:2]:
            for _, annotations in row.items():
                for annotation in annotations[:2]:
                    print(annotation)
        for row in fullSpless[:5]:
            for _, annotations in row.items():
                for annotation in annotations[:2]:
                    print(annotation.result)
        single = LightPipeline(loaded_model).annotate("Joe was running under the rain.")
        print(single)
        assert single["lemma"][2] == "run"

