/*
 * Copyright 2017-2021 John Snow Labs
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

package com.johnsnowlabs.nlp.annotators.ld.dl

import com.johnsnowlabs.nlp.annotators.sentence_detector_dl.SentenceDetectorDLModel
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.scalatest._

class LanguageDetectorDLTestSpec extends FlatSpec {

  val spark = ResourceHelper.spark

  "LanguageDetectorDL" should "correctly load pretrained model" taggedAs FastTest in {

    val smallCorpus = spark.read
      .option("header", true)
      .option("delimiter", "|")
      .csv("src/test/resources/language-detector/multilingual_sample.txt")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = SentenceDetectorDLModel.pretrained()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")

    val languageDetector = LanguageDetectorDL.pretrained()
      .setInputCols("sentence")
      .setOutputCol("language")

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentence,
        languageDetector
      ))

    pipeline.fit(smallCorpus).write.overwrite().save("./tmp_ld_pipeline")
    val pipelineModel = PipelineModel.load("./tmp_ld_pipeline")
    pipelineModel.transform(smallCorpus).select("language.result", "lang").show(2, false)

  }

}
