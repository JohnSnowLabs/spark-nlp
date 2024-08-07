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

package com.johnsnowlabs.nlp.annotators.classifier.dl

import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions.explode
import org.scalatest.flatspec.AnyFlatSpec

class CamemBertForZeroShotClassificationTestSpec extends AnyFlatSpec {

  "CamemBertForZeroShotClassification" should "correctly load custom ONNX model" taggedAs SlowTest in {
    import ResourceHelper.spark.implicits._

    val dataDf = Seq("L'équipe de France joue aujourd'hui au Parc des Princes").toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val zeroShotClassifier = CamemBertForZeroShotClassification
      .pretrained()
      .setOutputCol("multi_class")
      .setCaseSensitive(true)
      .setCoalesceSentences(true)
      .setCandidateLabels(Array("sport", "politique", "science"))

    val pipeline = new Pipeline().setStages(Array(document, tokenizer, zeroShotClassifier))

    val pipelineModel = pipeline.fit(dataDf)
    val pipelineDF = pipelineModel.transform(dataDf)

    pipelineDF.select("multi_class").show(false)
    val totalDocs = pipelineDF.select(explode($"document.result")).count.toInt
    val totalLabels = pipelineDF.select(explode($"multi_class.result")).count.toInt

    println(s"total tokens: $totalDocs")
    println(s"total labels: $totalLabels")

    assert(totalDocs == totalLabels)
  }

  it should "correctly load custom Tensorflow model" taggedAs SlowTest in {
    import ResourceHelper.spark.implicits._

    val dataDf = Seq("L'équipe de France joue aujourd'hui au Parc des Princes").toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val zeroShotClassifier = CamemBertForZeroShotClassification
      .pretrained("camembert-zero-shot-classifier-xnli-tf")
      .setOutputCol("multi_class")
      .setCaseSensitive(true)
      .setCoalesceSentences(true)
      .setCandidateLabels(Array("sport", "politique", "science"))

    val pipeline = new Pipeline().setStages(Array(document, tokenizer, zeroShotClassifier))

    val pipelineModel = pipeline.fit(dataDf)
    val pipelineDF = pipelineModel.transform(dataDf)

    pipelineDF.select("multi_class").show(false)
    val totalDocs = pipelineDF.select(explode($"document.result")).count.toInt
    val totalLabels = pipelineDF.select(explode($"multi_class.result")).count.toInt

    println(s"total tokens: $totalDocs")
    println(s"total labels: $totalLabels")

    assert(totalDocs == totalLabels)
  }

}
