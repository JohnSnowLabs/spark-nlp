package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorType.TOKEN
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{Annotation, DataBuilder}
import org.apache.spark.ml.Pipeline
import org.scalatest.FlatSpec

class RegexTokenizerTestSpec extends FlatSpec {

  "RegexTokenizer" should "correctly tokenize by space" in {

    val testData = ResourceHelper.spark.createDataFrame(Seq(
      (1, "This is my first sentence. This is my second."),
      (2, "This is my third sentence. This is my forth.")
    )).toDF("id", "text")

    val expectedTokens = Seq(
      Annotation(TOKEN, 0, 3, "this", Map("sentence" -> "0")),
      Annotation(TOKEN, 5, 6, "is", Map("sentence" -> "0")),
      Annotation(TOKEN, 8, 9, "my", Map("sentence" -> "0")),
      Annotation(TOKEN, 11, 15, "first", Map("sentence" -> "0")),
      Annotation(TOKEN, 17, 25, "sentence.", Map("sentence" -> "0")),
      Annotation(TOKEN, 0, 3, "this", Map("sentence" -> "1")),
      Annotation(TOKEN, 5, 6, "is", Map("sentence" -> "1")),
      Annotation(TOKEN, 8, 9, "my", Map("sentence" -> "1")),
      Annotation(TOKEN, 11, 17, "second.", Map("sentence" -> "1")),
      Annotation(TOKEN, 0, 3, "this", Map("sentence" -> "0")),
      Annotation(TOKEN, 5, 6, "is", Map("sentence" -> "0")),
      Annotation(TOKEN, 8, 9, "my", Map("sentence" -> "0")),
      Annotation(TOKEN, 11, 15, "third", Map("sentence" -> "0")),
      Annotation(TOKEN, 17, 25, "sentence.", Map("sentence" -> "0")),
      Annotation(TOKEN, 0, 3, "this", Map("sentence" -> "1")),
      Annotation(TOKEN, 5, 6, "is", Map("sentence" -> "1")),
      Annotation(TOKEN, 8, 9, "my", Map("sentence" -> "1")),
      Annotation(TOKEN, 11, 16, "forth.", Map("sentence" -> "1"))
    )

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val regexTokenizer = new RegexTokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("regexToken")
      .setToLowercase(true)
      .setPattern("\\s+")

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentence,
        regexTokenizer
      ))

    val pipelineDF = pipeline.fit(testData).transform(testData)

    //    pipelineDF.select(size(pipelineDF("regexToken.result")).as("totalTokens")).show
    //    pipelineDF.select(pipelineDF("document")).show(false)
    //    pipelineDF.select(pipelineDF("sentence")).show(false)
    //    pipelineDF.select(pipelineDF("regexToken.result")).show(false)
    //    pipelineDF.select(pipelineDF("regexToken")).show(false)

    val regexTokensResults = Annotation.collect(pipelineDF, "regexToken").flatten.toSeq
    assert(regexTokensResults == expectedTokens)

  }

  "RegexTokenizer" should "correctly tokenize by patterns" in {

    val testData = ResourceHelper.spark.createDataFrame(Seq(
      (1, "T1-T2 DATE**[12/24/13] 10/12, ph+ 90%"))).toDF("id", "text")

    val expectedTokens = Seq(
      Annotation(TOKEN, 0, 1, "t1", Map("sentence" -> "0")),
      Annotation(TOKEN, 3, 4, "t2", Map("sentence" -> "0")),
      Annotation(TOKEN, 6, 9, "date", Map("sentence" -> "0")),
      Annotation(TOKEN, 12, 21, "[12/24/13]", Map("sentence" -> "0")),
      Annotation(TOKEN, 23, 27, "10/12", Map("sentence" -> "0")),
      Annotation(TOKEN, 30, 32, "ph+", Map("sentence" -> "0")),
      Annotation(TOKEN, 34, 36, "90%", Map("sentence" -> "0"))
    )

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val regexTokenizer = new RegexTokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("regexToken")
      .setToLowercase(true)
      .setPattern("([^a-zA-Z\\/0-9\\[\\]+%])")

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentence,
        regexTokenizer
      ))

    val pipelineDF = pipeline.fit(testData).transform(testData)

    //    pipelineDF.select(size(pipelineDF("regexToken.result")).as("totalTokens")).show
    //    pipelineDF.select(pipelineDF("document")).show(false)
    //    pipelineDF.select(pipelineDF("sentence")).show(false)
    //    pipelineDF.select(pipelineDF("regexToken.result")).show(false)
    //    pipelineDF.select(pipelineDF("regexToken")).show(false)

    val regexTokensResults = Annotation.collect(pipelineDF, "regexToken").flatten.toSeq
    assert(regexTokensResults == expectedTokens)

  }

  "a Tokenizer" should "should correctly tokenize a parsed doc" in {

    val content = "1. T1-T2 DATE**[12/24/13] $1.99 () (10/12), ph+ 90%"
    val pattern = "\\s+|(?=[-.:;*+,$&%\\[\\]])|(?<=[-.:;*+,$&%\\[\\]])"

    val data = DataBuilder.basicDataBuild(content)

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetect = new SentenceDetector()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")
      .setCustomBounds(Array("\n"))

    val tokenizer = new RegexTokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("regexToken")
      .setPattern(pattern)
      .setPositionalMask(true)

    //#.setSplitPattern("\s+|(?=[^a-zA-Z0-9_/])|(?<=[^a-zA-Z0-9_/])")
    //#.setSplitPattern("\s+|(?=[-.:;*+,$&%\[\]])|(?<=[-.:;*+,$&%\[\]])")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetect, tokenizer))

    val pipelineDF = pipeline.fit(data).transform(data)

    //    pipelineDF.select("token").collect().foreach {
    //      row => println(row.getSeq[Row](0).map(Annotation(_)).mkString("\n"))
    //    }

    val expectedTokens = Seq(
      Annotation(TOKEN, 0, 0, "1", Map("sentence" -> "0")),
      Annotation(TOKEN, 1, 1, ".", Map("sentence" -> "0")),
      Annotation(TOKEN, 3, 4, "T1", Map("sentence" -> "0")),
      Annotation(TOKEN, 5, 5, "-", Map("sentence" -> "0")),
      Annotation(TOKEN, 6, 7, "T2", Map("sentence" -> "0")),
      Annotation(TOKEN, 9, 12, "DATE", Map("sentence" -> "0")),
      Annotation(TOKEN, 13, 13, "*", Map("sentence" -> "0")),
      Annotation(TOKEN, 14, 14, "*", Map("sentence" -> "0")),
      Annotation(TOKEN, 15, 15, "[", Map("sentence" -> "0")),
      Annotation(TOKEN, 16, 23, "12/24/13", Map("sentence" -> "0")),
      Annotation(TOKEN, 24, 24, "]", Map("sentence" -> "0")),
      Annotation(TOKEN, 26, 26, "$", Map("sentence" -> "0")),
      Annotation(TOKEN, 27, 27, "1", Map("sentence" -> "0")),
      Annotation(TOKEN, 28, 28, ".", Map("sentence" -> "0")),
      Annotation(TOKEN, 29, 30, "99", Map("sentence" -> "0")),
      Annotation(TOKEN, 32, 33, "()", Map("sentence" -> "0")),
      Annotation(TOKEN, 35, 41, "(10/12)", Map("sentence" -> "0")),
      Annotation(TOKEN, 42, 42, ",", Map("sentence" -> "0")),
      Annotation(TOKEN, 44, 45, "ph", Map("sentence" -> "0")),
      Annotation(TOKEN, 46, 46, "+", Map("sentence" -> "0")),
      Annotation(TOKEN, 48, 49, "90", Map("sentence" -> "0")),
      Annotation(TOKEN, 50, 50, "%", Map("sentence" -> "0"))
    )

    val regexTokensResults = Annotation.collect(pipelineDF, "regexToken").flatten.toSeq
    assert(regexTokensResults == expectedTokens)
  }

}
