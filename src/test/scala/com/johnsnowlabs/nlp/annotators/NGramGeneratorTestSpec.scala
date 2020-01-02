package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.AnnotatorType.CHUNK
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.ml.Pipeline
import org.scalatest.FlatSpec

class NGramGeneratorTestSpec extends FlatSpec {

  val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

  val sentence = new SentenceDetector()
    .setInputCols("document")
    .setOutputCol("sentence")

  val tokenizer = new Tokenizer()
    .setInputCols(Array("sentence"))
    .setOutputCol("token")

  "NGramGenerator" should "correctly generate n-grams from tokenizer's results" in {

    val testData = ResourceHelper.spark.createDataFrame(Seq(
      (1, "This is my first sentence. This is my second."),
      (2, "This is my third sentence. This is my fourth.")
    )).toDF("id", "text")

    val expectedNGrams = Seq(
      Annotation(CHUNK, 0, 6, "This is", Map("sentence" -> "0")),
      Annotation(CHUNK, 5, 9, "is my", Map("sentence" -> "0")),
      Annotation(CHUNK, 8, 15, "my first", Map("sentence" -> "0")),
      Annotation(CHUNK, 11, 24, "first sentence", Map("sentence" -> "0")),
      Annotation(CHUNK, 17, 25, "sentence .", Map("sentence" -> "0")),
      Annotation(CHUNK, 27, 33, "This is", Map("sentence" -> "1")),
      Annotation(CHUNK, 32, 36, "is my", Map("sentence" -> "1")),
      Annotation(CHUNK, 35, 43, "my second", Map("sentence" -> "1")),
      Annotation(CHUNK, 38, 44, "second .", Map("sentence" -> "1")),
      Annotation(CHUNK, 0, 6, "This is", Map("sentence" -> "0")),
      Annotation(CHUNK, 5, 9, "is my", Map("sentence" -> "0")),
      Annotation(CHUNK, 8, 15, "my third", Map("sentence" -> "0")),
      Annotation(CHUNK, 11, 24, "third sentence", Map("sentence" -> "0")),
      Annotation(CHUNK, 17, 25, "sentence .", Map("sentence" -> "0")),
      Annotation(CHUNK, 27, 33, "This is", Map("sentence" -> "1")),
      Annotation(CHUNK, 32, 36, "is my", Map("sentence" -> "1")),
      Annotation(CHUNK, 35, 43, "my fourth", Map("sentence" -> "1")),
      Annotation(CHUNK, 38, 44, "fourth .", Map("sentence" -> "1"))
    )

    val nGrams = new NGramGenerator()
      .setInputCols("token")
      .setOutputCol("ngrams")
      .setN(2)

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentence,
        tokenizer,
        nGrams
      ))

    val pipelineDF = pipeline.fit(testData).transform(testData)
    pipelineDF.select("token").show(false)
    pipelineDF.select("ngrams").show(false)

    val nGramGeneratorResults = Annotation.collect(pipelineDF, "ngrams").flatten.toSeq

    assert(nGramGeneratorResults == expectedNGrams)

  }

  "NGramGenerator" should "correctly generate n-grams with enableCumulative" in {

    val testData = ResourceHelper.spark.createDataFrame(Seq(
      (1, "This is my first sentence. This is my second."),
      (2, "This is my third sentence. This is my fourth.")
    )).toDF("id", "text")

    val expectedNGrams = Seq(
      Annotation(CHUNK, 0, 3, "This", Map("sentence" -> "0")),
      Annotation(CHUNK, 5, 6, "is", Map("sentence" -> "0")),
      Annotation(CHUNK, 8, 9, "my", Map("sentence" -> "0")),
      Annotation(CHUNK, 11, 15, "first", Map("sentence" -> "0")),
      Annotation(CHUNK, 17, 24, "sentence", Map("sentence" -> "0")),
      Annotation(CHUNK, 25, 25, ".", Map("sentence" -> "0")),
      Annotation(CHUNK, 0, 6, "This is", Map("sentence" -> "0")),
      Annotation(CHUNK, 5, 9, "is my", Map("sentence" -> "0")),
      Annotation(CHUNK, 8, 15, "my first", Map("sentence" -> "0")),
      Annotation(CHUNK, 11, 24, "first sentence", Map("sentence" -> "0")),
      Annotation(CHUNK, 17, 25, "sentence .", Map("sentence" -> "0")),
      Annotation(CHUNK, 27, 30, "This", Map("sentence" -> "1")),
      Annotation(CHUNK, 32, 33, "is", Map("sentence" -> "1")),
      Annotation(CHUNK, 35, 36, "my", Map("sentence" -> "1")),
      Annotation(CHUNK, 38, 43, "second", Map("sentence" -> "1")),
      Annotation(CHUNK, 44, 44, ".", Map("sentence" -> "1")),
      Annotation(CHUNK, 27, 33, "This is", Map("sentence" -> "1")),
      Annotation(CHUNK, 32, 36, "is my", Map("sentence" -> "1")),
      Annotation(CHUNK, 35, 43, "my second", Map("sentence" -> "1")),
      Annotation(CHUNK, 38, 44, "second .", Map("sentence" -> "1")),
      Annotation(CHUNK, 0, 3, "This", Map("sentence" -> "0")),
      Annotation(CHUNK, 5, 6, "is", Map("sentence" -> "0")),
      Annotation(CHUNK, 8, 9, "my", Map("sentence" -> "0")),
      Annotation(CHUNK, 11, 15, "third", Map("sentence" -> "0")),
      Annotation(CHUNK, 17, 24, "sentence", Map("sentence" -> "0")),
      Annotation(CHUNK, 25, 25, ".", Map("sentence" -> "0")),
      Annotation(CHUNK, 0, 6, "This is", Map("sentence" -> "0")),
      Annotation(CHUNK, 5, 9, "is my", Map("sentence" -> "0")),
      Annotation(CHUNK, 8, 15, "my third", Map("sentence" -> "0")),
      Annotation(CHUNK, 11, 24, "third sentence", Map("sentence" -> "0")),
      Annotation(CHUNK, 17, 25, "sentence .", Map("sentence" -> "0")),
      Annotation(CHUNK, 27, 30, "This", Map("sentence" -> "1")),
      Annotation(CHUNK, 32, 33, "is", Map("sentence" -> "1")),
      Annotation(CHUNK, 35, 36, "my", Map("sentence" -> "1")),
      Annotation(CHUNK, 38, 43, "fourth", Map("sentence" -> "1")),
      Annotation(CHUNK, 44, 44, ".", Map("sentence" -> "1")),
      Annotation(CHUNK, 27, 33, "This is", Map("sentence" -> "1")),
      Annotation(CHUNK, 32, 36, "is my", Map("sentence" -> "1")),
      Annotation(CHUNK, 35, 43, "my fourth", Map("sentence" -> "1")),
      Annotation(CHUNK, 38, 44, "fourth .", Map("sentence" -> "1"))
    )

    val nGrams = new NGramGenerator()
      .setInputCols("token")
      .setOutputCol("ngrams")
      .setN(2)
      .setEnableCumulative(true)

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentence,
        tokenizer,
        nGrams
      ))

    val pipelineDF = pipeline.fit(testData).transform(testData)
    pipelineDF.select("token").show(false)
    pipelineDF.select("ngrams").show(false)

    val nGramGeneratorResults = Annotation.collect(pipelineDF, "ngrams").flatten.toSeq

    assert(nGrams.getEnableCumulative == true)
    assert(nGramGeneratorResults == expectedNGrams)

  }

  "NGramGenerator" should "correctly generate n-grams with specified delimiter" in {
    val delimiter = "_"

    val testData = ResourceHelper.spark.createDataFrame(Seq(
      (1, "This is my first sentence. This is my second."),
      (2, "This is my third sentence. This is my fourth.")
    )).toDF("id", "text")

    val expectedNGrams = Seq(
      Annotation(CHUNK, 0, 3, "This", Map("sentence" -> "0")),
      Annotation(CHUNK, 5, 6, "is", Map("sentence" -> "0")),
      Annotation(CHUNK, 8, 9, "my", Map("sentence" -> "0")),
      Annotation(CHUNK, 11, 15, "first", Map("sentence" -> "0")),
      Annotation(CHUNK, 17, 24, "sentence", Map("sentence" -> "0")),
      Annotation(CHUNK, 25, 25, ".", Map("sentence" -> "0")),
      Annotation(CHUNK, 0, 6, "This_is", Map("sentence" -> "0")),
      Annotation(CHUNK, 5, 9, "is_my", Map("sentence" -> "0")),
      Annotation(CHUNK, 8, 15, "my_first", Map("sentence" -> "0")),
      Annotation(CHUNK, 11, 24, "first_sentence", Map("sentence" -> "0")),
      Annotation(CHUNK, 17, 25, "sentence_.", Map("sentence" -> "0")),
      Annotation(CHUNK, 27, 30, "This", Map("sentence" -> "1")),
      Annotation(CHUNK, 32, 33, "is", Map("sentence" -> "1")),
      Annotation(CHUNK, 35, 36, "my", Map("sentence" -> "1")),
      Annotation(CHUNK, 38, 43, "second", Map("sentence" -> "1")),
      Annotation(CHUNK, 44, 44, ".", Map("sentence" -> "1")),
      Annotation(CHUNK, 27, 33, "This_is", Map("sentence" -> "1")),
      Annotation(CHUNK, 32, 36, "is_my", Map("sentence" -> "1")),
      Annotation(CHUNK, 35, 43, "my_second", Map("sentence" -> "1")),
      Annotation(CHUNK, 38, 44, "second_.", Map("sentence" -> "1")),
      Annotation(CHUNK, 0, 3, "This", Map("sentence" -> "0")),
      Annotation(CHUNK, 5, 6, "is", Map("sentence" -> "0")),
      Annotation(CHUNK, 8, 9, "my", Map("sentence" -> "0")),
      Annotation(CHUNK, 11, 15, "third", Map("sentence" -> "0")),
      Annotation(CHUNK, 17, 24, "sentence", Map("sentence" -> "0")),
      Annotation(CHUNK, 25, 25, ".", Map("sentence" -> "0")),
      Annotation(CHUNK, 0, 6, "This_is", Map("sentence" -> "0")),
      Annotation(CHUNK, 5, 9, "is_my", Map("sentence" -> "0")),
      Annotation(CHUNK, 8, 15, "my_third", Map("sentence" -> "0")),
      Annotation(CHUNK, 11, 24, "third_sentence", Map("sentence" -> "0")),
      Annotation(CHUNK, 17, 25, "sentence_.", Map("sentence" -> "0")),
      Annotation(CHUNK, 27, 30, "This", Map("sentence" -> "1")),
      Annotation(CHUNK, 32, 33, "is", Map("sentence" -> "1")),
      Annotation(CHUNK, 35, 36, "my", Map("sentence" -> "1")),
      Annotation(CHUNK, 38, 43, "fourth", Map("sentence" -> "1")),
      Annotation(CHUNK, 44, 44, ".", Map("sentence" -> "1")),
      Annotation(CHUNK, 27, 33, "This_is", Map("sentence" -> "1")),
      Annotation(CHUNK, 32, 36, "is_my", Map("sentence" -> "1")),
      Annotation(CHUNK, 35, 43, "my_fourth", Map("sentence" -> "1")),
      Annotation(CHUNK, 38, 44, "fourth_.", Map("sentence" -> "1"))
    )

    val nGrams = new NGramGenerator()
      .setInputCols("token")
      .setOutputCol("ngrams")
      .setN(2)
      .setEnableCumulative(true)
      .setDelimiter(delimiter)

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentence,
        tokenizer,
        nGrams
      ))

    val pipelineDF = pipeline.fit(testData).transform(testData)
    pipelineDF.select("token").show(false)
    pipelineDF.select("ngrams").show(false)

    val nGramGeneratorResults = Annotation.collect(pipelineDF, "ngrams").flatten.toSeq
    assert(nGrams.getDelimiter == delimiter)
    assert(nGramGeneratorResults == expectedNGrams)

  }
}




