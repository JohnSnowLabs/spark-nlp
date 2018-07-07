package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.annotators.Tokenizer
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.StopWordsRemover
import org.scalatest._

class FinisherTestSpec extends FlatSpec {

  val data = ContentProvider.parquetData.limit(5)
  import data.sparkSession.implicits._

  val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

  val tokenizer = new Tokenizer()
    .setInputCols(Array("document"))
    .setOutputCol("token")

  "A Finisher with default settings" should "return clean results" in {

    val finisher = new Finisher()
      .setInputCols("token")
      .setOutputAsArray(false)
      .setAnnotationSplitSymbol("@")
      .setValueSplitSymbol("#")

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        tokenizer,
        finisher
      ))

    val result = pipeline.fit(data).transform(data)

    result.show()
    assert(result.columns.length == 4, "because finisher did not clean annotations or did not return proper columns")
    result.select("finished_token").as[String].collect.foreach(s => assert(s.contains("@"), "because @ separator string was not found"))
  }

  "A Finisher with custom settings" should "behave accordingly" in {

    val finisher = new Finisher()
      .setInputCols("token")
      .setOutputCols("token_out")
      .setOutputAsArray(false)
      .setAnnotationSplitSymbol("%")
      .setValueSplitSymbol("&")
      .setCleanAnnotations(false)
      .setIncludeMetadata(true)

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        tokenizer,
        finisher
      ))

    val result = pipeline.fit(data).transform(data)

    result.show()
    assert(result.columns.length == 6, "because finisher removed annotations or did not return proper columns")
    assert(result.columns.contains("token_out"))
    result.select("token_out").as[String].collect.foreach(s => assert(s.contains("%"), "because % separator string was not found"))
    result.select("token_out").as[String].collect.foreach(s => assert(s.contains("->"), "because -> key value was not found"))

  }

  "A Finisher with array output" should "behave accordingly with SparkML StopWords" in {

    val finisher = new Finisher()
      .setInputCols("token")
      .setOutputCols("token_out")
      .setOutputAsArray(true)

    val stopWords = new StopWordsRemover()
      .setInputCol("token_out")
      .setOutputCol("stopped")

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        tokenizer,
        finisher,
        stopWords
      ))

    val result = pipeline.fit(data).transform(data)

    result.show()
    assert(result.columns.length == 5, "because finisher removed annotations or did not return proper columns")
    assert(result.columns.contains("stopped"))

  }

}
