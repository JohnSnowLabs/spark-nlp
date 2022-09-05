/*
 * Copyright 2017-2022 John Snow Labs
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

package com.johnsnowlabs.nlp.annotators.audio

import com.johnsnowlabs.nlp.AudioAssembler
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.flatspec.AnyFlatSpec

class Wav2Vec2ForCTCTestSpec extends AnyFlatSpec {
  val spark: SparkSession = ResourceHelper.spark

  val wavPath = "src/test/resources/audio/wav/"

  val wavDf: DataFrame = spark.read
    .format("binaryFile")
    .load(wavPath)
    .repartition(1)

//  wavDf.show()

  val audioAssembler: AudioAssembler = new AudioAssembler()
    .setInputCol("content")
    .setOutputCol("audio_assembler")

  val speechToText: Wav2Vec2ForCTC = Wav2Vec2ForCTC
//    .pretrained()
    .loadSavedModel("/Users/maziyar/Downloads/export_wav2vec2-base-960h", ResourceHelper.spark)
    .setInputCols("audio_assembler")
    .setOutputCol("text")
    .setBatchSize(1)

  val pipeline: Pipeline = new Pipeline().setStages(Array(audioAssembler, speechToText))

  "Wav2Vec2ForCTC" should "correctly transform speech to text" taggedAs SlowTest in {

    val pipelineDF = pipeline.fit(wavDf).transform(wavDf)

    pipelineDF.select("text").show(10, false)

  }

}
