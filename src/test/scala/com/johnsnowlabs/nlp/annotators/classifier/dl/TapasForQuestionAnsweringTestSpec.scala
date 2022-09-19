package com.johnsnowlabs.nlp.annotators.classifier.dl

import com.johnsnowlabs.nlp.annotators.sentence_detector_dl.SentenceDetectorDLModel
import com.johnsnowlabs.nlp.base.MultiDocumentAssembler
import com.johnsnowlabs.nlp.{DocumentAssembler, TableAssembler}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

import scala.io.Source

class TapasForQuestionAnsweringTestSpec extends AnyFlatSpec {
  import ResourceHelper.spark.implicits._

  "TapasForQuestionAnswering" should "load saved model" taggedAs SlowTest ignore {
    TapasForQuestionAnswering
      .loadSavedModel("/tmp/tapas_tf", ResourceHelper.spark)
      .save("/models/sparknlp/tapas")
  }

  "TapasForQuestionAnswering" should "prepare inputs" in {
    val sourceFile = Source.fromFile("src/test/resources/tapas/rich_people.json")
    val textFileContents = sourceFile.getLines().mkString("\n")
    sourceFile.close()

    val data = Seq(
      (textFileContents, "Who is the richest man? Who is poorer?"),
    ).toDF("table", "questions").repartition(1)

    val docAssembler = new MultiDocumentAssembler()
      .setInputCols("table", "questions")
      .setOutputCols("document_table", "document_questions")

    val sentenceDetector = SentenceDetectorDLModel.pretrained()
      .setInputCols(Array("document_questions"))
      .setOutputCol("question")

    val tableAssembler = new TableAssembler()
      .setInputCols(Array("document_table"))
      .setOutputCol("table")

    val tapas = TapasForQuestionAnswering
      .load("/models/sparknlp/tapas")
      .setCaseSensitive(false)
      .setMaxSentenceLength(512)
      .setInputCols(Array("question", "table"))
      .setOutputCol("answer")

    val pipeline = new Pipeline().setStages(Array(docAssembler, sentenceDetector, tableAssembler, tapas))
    val pipelineModel = pipeline.fit(data)
    pipelineModel.transform(data).collect()

  }
}
