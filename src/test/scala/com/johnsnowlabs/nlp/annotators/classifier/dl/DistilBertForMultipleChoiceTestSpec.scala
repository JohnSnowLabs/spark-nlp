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

import com.johnsnowlabs.nlp.{Annotation, AssertAnnotations, MultiDocumentAssembler}
import com.johnsnowlabs.nlp.annotators.SparkSessionTest
import com.johnsnowlabs.nlp.base.LightPipeline
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.scalatest.flatspec.AnyFlatSpec

class DistilBertForMultipleChoiceTestSpec extends AnyFlatSpec with SparkSessionTest {

  import spark.implicits._

  lazy val pipelineModel = getDistilBertForMultipleChoicePipelineModel

  val testDataframe =
    Seq(("The Eiffel Tower is located in which country?", "Germany, France, Italy"))
      .toDF("question", "context")

  "DistilBertForMultipleChoiceTestSpec" should "answer a multiple choice question" taggedAs SlowTest in {
    val resultDf = pipelineModel.transform(testDataframe)
    resultDf.show(truncate = false)

    val result = AssertAnnotations.getActualResult(resultDf, "answer")
    result.foreach { annotation =>
      annotation.foreach(a => assert(a.result.nonEmpty))
    }
  }

  it should "work with light pipeline fullAnnotate" taggedAs SlowTest in {
    val lightPipeline = new LightPipeline(pipelineModel)
    val resultFullAnnotate = lightPipeline.fullAnnotate(
      "The Eiffel Tower is located in which country?",
      "Germany, France, Italy")
    println(s"resultAnnotate: $resultFullAnnotate")

    val answerAnnotation = resultFullAnnotate("answer").head.asInstanceOf[Annotation]

    assert(answerAnnotation.result.nonEmpty)
  }

  it should "work with light pipeline annotate" taggedAs SlowTest in {
    val lightPipeline = new LightPipeline(pipelineModel)
    val resultAnnotate = lightPipeline.annotate(
      "The Eiffel Tower is located in which country?",
      "Germany, France, Italy")
    println(s"resultAnnotate: $resultAnnotate")

    assert(resultAnnotate("answer").head.nonEmpty)
  }

  private def getDistilBertForMultipleChoicePipelineModel: PipelineModel = {
    val documentAssembler = new MultiDocumentAssembler()
      .setInputCols("question", "context")
      .setOutputCols("document_question", "document_context")

    val bertForMultipleChoice = DistilBertForMultipleChoice
      .pretrained()
      .setInputCols("document_question", "document_context")
      .setOutputCol("answer")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, bertForMultipleChoice))

    pipeline.fit(emptyDataSet)
  }

}
