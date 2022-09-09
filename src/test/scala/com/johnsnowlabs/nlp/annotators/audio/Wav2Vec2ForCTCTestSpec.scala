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
import org.apache.spark.sql.SparkSession
import org.scalatest.flatspec.AnyFlatSpec

class Wav2Vec2ForCTCTestSpec extends AnyFlatSpec {

  val spark: SparkSession = ResourceHelper.spark
  import spark.implicits._

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

  "Wav2Vec2ForCTC" should "correctly transform speech to text from already processed audio files" taggedAs SlowTest in {

    val bufferedSource =
      scala.io.Source.fromFile("src/test/resources/audio/csv/audi_floats.csv")

    val rawFloats = bufferedSource
      .getLines()
      .map(_.split(",").head.trim.toFloat)
      .toArray
    bufferedSource.close

    val rawDF = Seq(rawFloats).toDF("content")
    rawDF.printSchema()

    val pipelineDF = pipeline.fit(rawDF).transform(rawDF)

    pipelineDF.select("text").show(10, false)

  }

}
