/*
 * Copyright 2017-2023 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.annotators.seq2seq

import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class LLAMA3TestSpec extends AnyFlatSpec {

  "llama3" should "should handle temperature=0 correctly and not crash when predicting more than 1 element with doSample=True" taggedAs SlowTest in {
    // Even tough the Paper states temperature in interval [0,1), using temperature=0 will result in division by 0 error.
    // Also DoSample=True may result in infinities being generated and distFiltered.length==0 which results in exception if we don't return 0 instead internally.
    val testData = ResourceHelper.spark
      .createDataFrame(
        Seq(
          (
            1,
            """<|start_header_id|>system<|end_header_id|>

          You are a minion chatbot who always responds in minion speak!

          <|start_header_id|>user<|end_header_id|>

          Who are you?

          <|start_header_id|>assistant<|end_header_id|>
          """.stripMargin)))
      .toDF("id", "text")
      .repartition(1)
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("documents")

    val llama3 = LLAMA3Transformer
      .pretrained()
      .setInputCols(Array("documents"))
      .setDoSample(true)
      .setMaxOutputLength(50)
      .setOutputCol("generation")
      .setBeamSize(4)
      .setStopTokenIds(Array(128001))
      .setTemperature(0.6)
      .setTopP(0.9)
      .setTopK(-1)
    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, llama3))

    val pipelineModel = pipeline.fit(testData)

    pipelineModel
      .transform(testData)
      .show(truncate = false)

    pipelineModel
      .transform(testData)
      .show(truncate = false)

    pipelineModel.stages.last
      .asInstanceOf[LLAMA3Transformer]
      .write
      .overwrite()
      .save("/tmp/llama3-7b-4bit-model")

    val loadedLLAMA3 = LLAMA3Transformer.load("/tmp/llama3-7b-4bit-model")

    val loadedPipeline = new Pipeline().setStages(Array(documentAssembler, loadedLLAMA3))

    loadedPipeline
      .fit(testData)
      .transform(testData)
      .show(truncate = false)

  }
}
