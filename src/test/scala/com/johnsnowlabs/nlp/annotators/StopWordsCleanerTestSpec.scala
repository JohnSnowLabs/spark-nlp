package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.AnnotatorType.TOKEN
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions.size
import org.scalatest.FlatSpec

class StopWordsCleanerTestSpec extends FlatSpec {
  val documentAssembler: DocumentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

  val sentence: SentenceDetector = new SentenceDetector()
    .setInputCols("document")
    .setOutputCol("sentence")

  val tokenizer: Tokenizer = new Tokenizer()
    .setInputCols(Array("sentence"))
    .setOutputCol("token")

  "StopWordsCleaner" should "correctly remove stop words from tokenizer's results" taggedAs FastTest in {

    val testData = ResourceHelper.spark.createDataFrame(Seq(
      (1, "This is my first sentence. This is my second."),
      (2, "This is my third sentence. This is my forth.")
    )).toDF("id", "text")

    // Let's remove "this" and "is" as stop words
    val expectedWithoutStopWords = Seq(
      Annotation(TOKEN, 8, 9, "my", Map("sentence" -> "0")),
      Annotation(TOKEN, 11, 15, "first", Map("sentence" -> "0")),
      Annotation(TOKEN, 17, 24, "sentence", Map("sentence" -> "0")),
      Annotation(TOKEN, 25, 25, ".", Map("sentence" -> "0")),
      Annotation(TOKEN, 35, 36, "my", Map("sentence" -> "1")),
      Annotation(TOKEN, 38, 43, "second", Map("sentence" -> "1")),
      Annotation(TOKEN, 44, 44, ".", Map("sentence" -> "1")),
      Annotation(TOKEN, 8, 9, "my", Map("sentence" -> "0")),
      Annotation(TOKEN, 11, 15, "third", Map("sentence" -> "0")),
      Annotation(TOKEN, 17, 24, "sentence", Map("sentence" -> "0")),
      Annotation(TOKEN, 25, 25, ".", Map("sentence" -> "0")),
      Annotation(TOKEN, 35, 36, "my", Map("sentence" -> "1")),
      Annotation(TOKEN, 38, 42, "forth", Map("sentence" -> "1")),
      Annotation(TOKEN, 43, 43, ".", Map("sentence" -> "1"))
    )

    val stopWords = new StopWordsCleaner()
      .setInputCols("token")
      .setOutputCol("cleanTokens")
      .setStopWords(Array("this", "is"))
      .setCaseSensitive(false)

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentence,
        tokenizer,
        stopWords
      ))

    val pipelineDF = pipeline.fit(testData).transform(testData)

    pipelineDF.select(size(pipelineDF("token.result")).as("totalTokens")).show
    pipelineDF.select(size(pipelineDF("cleanTokens.result")).as("totalCleanedTokens")).show

    val tokensWithoutStopWords = Annotation.collect(pipelineDF, "cleanTokens").flatten.toSeq

    assert(tokensWithoutStopWords == expectedWithoutStopWords)

  }

  "StopWordsCleaner" should "successfully downloads pretrained models" taggedAs FastTest in {

    val testData = ResourceHelper.spark.createDataFrame(Seq(
      (1, "This is my first sentence. This is my second."),
      (2, "This is my third sentence. This is my forth.")
    )).toDF("id", "text")

    val stopWords = StopWordsCleaner.pretrained("stopwords_en")
      .setInputCols("token")
      .setOutputCol("cleanTokens")
      .setCaseSensitive(false)

    //    stopWords.getStopWords.foreach(println)

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentence,
        tokenizer,
        stopWords
      ))

    pipeline.fit(testData).transform(testData)

  }
}
