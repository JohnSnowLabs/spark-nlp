package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.annotator.SentenceDetector
import com.johnsnowlabs.nlp.{DocumentAssembler, SparkAccessor}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession
import org.scalatest.{BeforeAndAfterAll, Suite}

trait SparkSessionTest extends BeforeAndAfterAll { this: Suite =>

  val spark: SparkSession = SparkAccessor.spark
  val tokenizerPipeline = new Pipeline()
  val tokenizerWithSentencePipeline = new Pipeline()
  val documentAssembler = new DocumentAssembler()

  override def beforeAll(): Unit = {
    super.beforeAll()

    documentAssembler.setInputCol("text").setOutputCol("document")
    val tokenizer = new Tokenizer()
    tokenizer.setInputCols("document").setOutputCol("token")
    tokenizerPipeline.setStages(Array(documentAssembler, tokenizer))

    val sentenceDetector = new SentenceDetector()
    sentenceDetector.setInputCols("document").setOutputCol("sentence")
    val tokenizerWithSentence = new Tokenizer()
    tokenizerWithSentence.setInputCols("sentence").setOutputCol("token")
    tokenizerWithSentencePipeline.setStages(Array(documentAssembler, sentenceDetector, tokenizerWithSentence))
  }

  override def afterAll(): Unit = {
    try super.afterAll()
    spark.stop()
  }

}
