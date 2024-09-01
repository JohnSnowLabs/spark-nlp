/*
 * Copyright 2017-2024 John Snow Labs
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
import org.sparkproject.dmg.pmml.True

class CPMTestSpec extends AnyFlatSpec {

  "minicpm" should "should handle temperature=0 correctly and not crash when predicting more than 1 element with doSample=True" taggedAs SlowTest in {
    // Even tough the Paper states temperature in interval [0,1), using temperature=0 will result in division by 0 error.
    // Also DoSample=True may result in infinities being generated and distFiltered.length==0 which results in exception if we don't return 0 instead internally.
    val testData = ResourceHelper.spark
      .createDataFrame(Seq((1, "My name is Leonardo.")))
      .toDF("id", "text")
      .repartition(1)
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("documents")

    val bart = CPMTransformer
      .pretrained()
      .setInputCols(Array("documents"))
      .setDoSample(true)
      .setTopP(0.8)
      .setTopK(100)
      .setTemperature(0.8)
      .setMaxOutputLength(50)
      .setOutputCol("generation")
      .setBeamSize(2)

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, bart))

    val pipelineModel = pipeline.fit(testData)

    pipelineModel
      .transform(testData)
      .show(truncate = false)

    pipelineModel.stages.last
      .asInstanceOf[CPMTransformer]
      .write
      .overwrite()
      .save("/tmp/minicpm-model")
  }
}
