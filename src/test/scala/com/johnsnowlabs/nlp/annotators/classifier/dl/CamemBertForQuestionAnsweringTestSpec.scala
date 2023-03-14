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

package com.johnsnowlabs.nlp.annotators.classifier.dl

import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class CamemBertForQuestionAnsweringTestSpec extends AnyFlatSpec {

  import ResourceHelper.spark.implicits._

  "CamemBertForQuestionAnswering" should "correctly load custom model with extracted signatures" taggedAs SlowTest in {

    val fquadContext =
      """L'idée selon laquelle une planète inconnue
        |pourrait exister entre les orbites de Mars et
        |Jupiter fut proposée pour la première fois par
        |Johann Elert Bode en 1768. Ses suggestions étaient
        |basées sur la loi de Titius-Bode, une théorie
        |désormais obsolète proposée par Johann Daniel
        |Titius en 1766,. Selon cette loi, le demi-grand
        |axe de cette planète aurait été d'environ 2,8 ua.
        |La découverte d'Uranus par William Herschel en
        |1781 accrut la confiance dans la loi de Titius-
        |Bode et, en 1800, vingt-quatre astronomes
        |expérimentés combinèrent leurs efforts et
        |entreprirent une recherche méthodique de la
        |planète proposée,. Le groupe était dirigé par
        |Franz Xaver von Zach. Bien qu'ils n'aient pas
        |découvert Cérès, ils trouvèrent néanmoins
        |plusieurs autres astéroïdes.""".stripMargin

    val ddd = Seq(
      (
        "Quel astronome a émit l'idée en premier d'une planète entre les orbites de Mars et Jupiter ?",
        fquadContext),
      ("Quel astronome découvrit Uranus ?", fquadContext),
      ("Quelles furent les découvertes finales des vingt-quatre astronomes ?", fquadContext),
      ("Où est-ce que je vis?", "Mon nom est Wolfgang et je vis à Berlin"))
      .toDF("question", "context")
      .repartition(1)

    val document = new MultiDocumentAssembler()
      .setInputCols("question", "context")
      .setOutputCols("document_question", "document_context")

    val questionAnswering = CamemBertForQuestionAnswering
      .pretrained()
      .setInputCols(Array("document_question", "document_context"))
      .setOutputCol("answer")
      .setCaseSensitive(true)
      .setMaxSentenceLength(512)

    val pipeline = new Pipeline().setStages(Array(document, questionAnswering))

    val pipelineModel = pipeline.fit(ddd)
    val pipelineDF = pipelineModel.transform(ddd)

    pipelineDF.show(false)
    pipelineDF.select("answer").show(false)
    pipelineDF.select("answer.result").show(false)

  }

  "CamemBertForQuestionAnswering" should "benchmark test" taggedAs SlowTest in {

    val data = ResourceHelper.spark.read
      .option("header", "true")
      .option("escape", "\"")
      .csv("src/test/resources/squad/validation-squad-sample.csv")
      .sample(0.1)

    println(data.count())
    data.show(5)

    val document = new MultiDocumentAssembler()
      .setInputCols("question", "context")
      .setOutputCols("document_question", "document_context")

    val questionAnswering = CamemBertForQuestionAnswering
      .pretrained()
      .setInputCols(Array("document_question", "document_context"))
      .setOutputCol("answer")
      .setCaseSensitive(true)
      .setMaxSentenceLength(512)

    val pipeline = new Pipeline().setStages(Array(document, questionAnswering))

    val pipelineModel = pipeline.fit(data)
    val pipelineDF = pipelineModel.transform(data)

    Benchmark.time("Time to show CamemBertForQuestionAnswering results") {
      pipelineDF.select("answer").show(10, false)
    }

    Benchmark.time("Time to save CamemBertForQuestionAnswering results") {
      pipelineDF
        .select("answer.result")
        .write
        .mode("overwrite")
        .parquet("./tmp_question_answering")
    }

  }
}
