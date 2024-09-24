#  Copyright 2017-2023 John Snow Labs
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
class AutoGGUFModelTestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkContextForTest.spark

    def runTest(self):
        data = (
            self.spark.createDataFrame(
                [
                    ["The moons of Jupiter are "],
                    ["Earth is "],
                    ["The moon is "],
                    ["The sun is "],
                ]
            )
            .toDF("text")
            .repartition(1)
        )

        document_assembler = (
            DocumentAssembler().setInputCol("text").setOutputCol("document")
        )

        model = (
            AutoGGUFModel.pretrained()
            .setInputCols("document")
            .setOutputCol("completions")
            .setBatchSize(4)
            .setNPredict(20)
            .setNGpuLayers(5)
            .setTemperature(0.4)
            .setTopK(40)
            .setTopP(0.9)
            .setPenalizeNl(True)
        )

        pipeline = Pipeline().setStages([document_assembler, model])
        results = pipeline.fit(data).transform(data)

        results.select("completions").show(truncate=False)


@pytest.mark.slow
class AutoGGUFModelParametersTestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkContextForTest.spark

    def runTest(self):
        data = (
            self.spark.createDataFrame([["The moons of Jupiter are "]])
            .toDF("text")
            .repartition(1)
        )

        document_assembler = (
            DocumentAssembler().setInputCol("text").setOutputCol("document")
        )

        model = (
            AutoGGUFModel.pretrained()
            .setInputCols("document")
            .setOutputCol("completions")
            .setBatchSize(4)
        )

        # Model Parameters
        model.setNThreads(8)
        model.setNThreadsDraft(8)
        model.setNThreadsBatch(8)
        model.setNThreadsBatchDraft(8)
        model.setNCtx(512)
        model.setNBatch(32)
        model.setNUbatch(32)
        model.setNDraft(5)
        model.setNChunks(-1)
        model.setNSequences(1)
        model.setPSplit(0.1)
        model.setNGpuLayers(99)
        model.setNGpuLayersDraft(99)
        model.setGpuSplitMode("NONE")
        model.setMainGpu(0)
        model.setTensorSplit([])
        model.setNBeams(0)
        model.setGrpAttnN(1)
        model.setGrpAttnW(512)
        model.setRopeFreqBase(1.0)
        model.setRopeFreqScale(1.0)
        model.setYarnExtFactor(1.0)
        model.setYarnAttnFactor(1.0)
        model.setYarnBetaFast(32.0)
        model.setYarnBetaSlow(1.0)
        model.setYarnOrigCtx(0)
        model.setDefragmentationThreshold(-1.0)
        model.setNumaStrategy("DISTRIBUTE")
        model.setRopeScalingType("UNSPECIFIED")
        model.setPoolingType("UNSPECIFIED")
        model.setModelDraft("")
        model.setLookupCacheStaticFilePath("/tmp/sparknlp-llama-cpp-cache")
        model.setLookupCacheDynamicFilePath("/tmp/sparknlp-llama-cpp-cache")
        model.setLoraBase("")
        model.setEmbedding(False)
        model.setFlashAttention(False)
        model.setInputPrefixBos(False)
        model.setUseMmap(False)
        model.setUseMlock(False)
        model.setNoKvOffload(False)
        model.setSystemPrompt("")
        model.setChatTemplate("")

        # Inference Parameters
        model.setInputPrefix("")
        model.setInputSuffix("")
        model.setCachePrompt(False)
        model.setNPredict(-1)
        model.setTopK(40)
        model.setTopP(0.9)
        model.setMinP(0.1)
        model.setTfsZ(1.0)
        model.setTypicalP(1.0)
        model.setTemperature(0.8)
        model.setDynamicTemperatureRange(0.0)
        model.setDynamicTemperatureExponent(1.0)
        model.setRepeatLastN(64)
        model.setRepeatPenalty(1.0)
        model.setFrequencyPenalty(0.0)
        model.setPresencePenalty(0.0)
        model.setMiroStat("DISABLED")
        model.setMiroStatTau(5.0)
        model.setMiroStatEta(0.1)
        model.setPenalizeNl(False)
        model.setNKeep(0)
        model.setSeed(-1)
        model.setNProbs(0)
        model.setMinKeep(0)
        model.setGrammar("")
        model.setPenaltyPrompt("")
        model.setIgnoreEos(False)
        model.setDisableTokenIds([])
        model.setStopStrings([])
        model.setUseChatTemplate(False)
        model.setNPredict(2)
        model.setSamplers(["TOP_P", "TOP_K"])

        # Special PySpark Parameters (Scala StructFeatures)
        model.setTokenIdBias({0: 0.0, 1: 0.0})
        model.setTokenBias({"!": 0.0, "?": 0.0})
        model.setLoraAdapters({" ": 0.0})

        pipeline = Pipeline().setStages([document_assembler, model])
        results = pipeline.fit(data).transform(data)

        results.select("completions").show(truncate=False)


@pytest.mark.slow
class AutoGGUFModelMetadataTestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkContextForTest.spark

    def runTest(self):
        model = (
            AutoGGUFModel.pretrained()
            .setInputCols("document")
            .setOutputCol("completions")
        )

        metadata = model.getMetadata()
        assert len(metadata) > 0
        print(eval(metadata))
