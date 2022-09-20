package com.johnsnowlabs.nlp.annotators.classifier.dl

import com.johnsnowlabs.nlp.annotators.classifier.dl.tapas.TapasForQuestionAnswering
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

  "TapasForQuestionAnswering" should "load saved model" taggedAs SlowTest in {
    TapasForQuestionAnswering
      .loadSavedModel("/tmp/tapas_tf", ResourceHelper.spark)
      .write.overwrite.save("/models/sparknlp/tapas")
  }

  "TapasForQuestionAnswering" should "prepare inputs" in {
    val sourceFile = Source.fromFile("src/test/resources/tapas/rich_people.json")
    val textFileContents = sourceFile.getLines().mkString("\n")
    sourceFile.close()

    val data = Seq(
      (textFileContents, "Who has more money? How much money has Donald Trump? How old are they?"),
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
    println(textFileContents)
    pipelineModel
      .transform(data)
      .selectExpr("explode(answer) as answer")
      .selectExpr("answer.metadata.question", "answer.result", "answer.metadata.cell_positions", "answer.metadata.cell_scores")
      .show(truncate = false)

  }
}
