package com.johnsnowlabs.nlp.custom.annotators

import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, TOKEN}
import com.johnsnowlabs.nlp.annotators.SparkSessionTest
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.{Annotation, Annotator, AssertAnnotations, LightPipeline}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.scalatest.flatspec.AnyFlatSpec

class CustomAnnotatorsTest extends AnyFlatSpec with SparkSessionTest {

  import spark.implicits._

  private val sentence1 = "In London, John Snow is a Physician."
  private val sentence2 = "In Castle Black, Jon Snow is a Lord Commander"
  private val text = sentence1 + " " + sentence2
  private val textDataSet = Seq(text).toDS.toDF("text")

  "UpperCase Custom Annotator" should "transform dataset to uppercase text" in {

    val pipelineModel = buildUpperCasePipelineModel
    val expectedAnnotation = Array(Seq(
      Annotation(DOCUMENT, 0, sentence1.length - 1, sentence1.toUpperCase(), Map("sentence" -> "0")),
      Annotation(DOCUMENT, sentence1.length + 1, sentence1.length + sentence2.length,
        sentence2.toUpperCase, Map("sentence" -> "1"))
    ))

    val resultDataSet = pipelineModel.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "sentence")

    AssertAnnotations.assertFields(expectedAnnotation, actualEntities)
  }

  it should "work with light pipeline" in {
    val pipelineModel = buildUpperCasePipelineModel
    val lightPipeline = new LightPipeline(pipelineModel)
    val expectedAnnotation = Seq(
      Annotation(DOCUMENT, 0, sentence1.length - 1, sentence1.toUpperCase(), Map("sentence" -> "0")),
      Annotation(DOCUMENT, sentence1.length + 1, sentence1.length + sentence2.length,
        sentence2.toUpperCase, Map("sentence" -> "1"))
    )

    val result = lightPipeline.fullAnnotate(text)

    assert(expectedAnnotation == result("sentence"))
  }

  private def buildUpperCasePipelineModel: PipelineModel = {
    val upperCase = new UpperCaseAnnotator()
      .setInputCols("document")
      .setOutputCol("upper")
    val sentenceDetector = new SentenceDetector()
      .setInputCols("upper")
      .setOutputCol("sentence")
    val pipeline = new Pipeline()
    pipeline.setStages(Array(documentAssembler, upperCase, sentenceDetector))
    val pipelineModel = pipeline.fit(emptyDataSet)
    pipelineModel
  }

  "StopWords Custom Annotator" should "work for input/output token annotators" in {
    val textDataSet = Seq(sentence1).toDS.toDF("text")
    val pipelineModel = buildStopWordsPipelineModel
    val expectedAnnotation = Array(Seq(
      Annotation(TOKEN, 0, 1, "In", Map("sentence" -> "0")),
      Annotation(TOKEN, 9, 9, ",", Map("sentence" -> "0")),
      Annotation(TOKEN, 21, 22, "is", Map("sentence" -> "0")),
      Annotation(TOKEN, 24, 24, "a", Map("sentence" -> "0")),
      Annotation(TOKEN, 35, 35, ".", Map("sentence" -> "0"))
    ))

    val resultDataSet = pipelineModel.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "stop_words")

    AssertAnnotations.assertFields(expectedAnnotation, actualEntities)
  }

  it should "work with light pipeline" in {
    val pipelineModel = buildStopWordsPipelineModel
    val lightPipeline = new LightPipeline(pipelineModel)
    val expectedAnnotation = Seq(
      Annotation(TOKEN, 0, 1, "In", Map("sentence" -> "0")),
      Annotation(TOKEN, 9, 9, ",", Map("sentence" -> "0")),
      Annotation(TOKEN, 21, 22, "is", Map("sentence" -> "0")),
      Annotation(TOKEN, 24, 24, "a", Map("sentence" -> "0")),
      Annotation(TOKEN, 35, 35, ".", Map("sentence" -> "0"))
    )

    val result = lightPipeline.fullAnnotate(sentence1)

    assert(expectedAnnotation == result("stop_words"))
  }

  private def buildStopWordsPipelineModel: PipelineModel = {
    val stopWords = new StopWordsAnnotator()
      .setInputCols("token")
      .setOutputCol("stop_words")
    val pipeline = new Pipeline()
    pipeline.setStages(Array(documentAssembler, tokenizer, stopWords))
    val pipelineModel = pipeline.fit(emptyDataSet)

    pipelineModel
  }

}

class UpperCaseAnnotator extends Annotator {

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    annotations.map(annotation =>
      Annotation(outputAnnotatorType, annotation.begin, annotation.end, annotation.result.toUpperCase(),
        annotation.metadata))
  }

}

class StopWordsAnnotator extends Annotator {

  override val inputAnnotatorTypes: Array[String] = Array(TOKEN)

  override val outputAnnotatorType: AnnotatorType = TOKEN

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    val tokens = annotations.filter(annotation => annotation.annotatorType == TOKEN)

    tokens.flatMap{ token =>
        if (token.result.length <= 2) {
          Some(Annotation(TOKEN, token.begin, token.end, token.result, token.metadata))
        } else None
    }
  }
}