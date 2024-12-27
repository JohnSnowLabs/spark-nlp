package com.johnsnowlabs.nlp.annotators.classifier.dl

import com.johnsnowlabs.nlp.{Annotation, AssertAnnotations, MultiDocumentAssembler}
import com.johnsnowlabs.nlp.annotators.SparkSessionTest
import com.johnsnowlabs.nlp.base.LightPipeline
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class AlbertForMultipleChoiceTest extends AnyFlatSpec with SparkSessionTest {

  import spark.implicits._

  lazy val pipelineModel = getAlbertForMultipleChoicePipelineModel

  val testDataframe =
    Seq(("The Eiffel Tower is located in which country?", "Germany, France, Italy"))
      .toDF("question", "context")

  "AlbertForMultipleChoice" should "answer a multiple choice question" taggedAs SlowTest in {
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

  private def getAlbertForMultipleChoicePipelineModel = {
    val documentAssembler = new MultiDocumentAssembler()
      .setInputCols("question", "context")
      .setOutputCols("document_question", "document_context")

    val bertForMultipleChoice = AlbertForMultipleChoice
      .pretrained()
      .setInputCols("document_question", "document_context")
      .setOutputCol("answer")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, bertForMultipleChoice))

    pipeline.fit(emptyDataSet)
  }

}
