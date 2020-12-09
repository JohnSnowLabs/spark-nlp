package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.{Annotation, DataBuilder}
import com.johnsnowlabs.nlp.AnnotatorType.TOKEN
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions.size
import org.scalatest.FlatSpec

class RegexTokenizerTestSpec extends FlatSpec {

//  "RegexTokenizer" should "correctly tokenize by space" in {
//
//    val testData = ResourceHelper.spark.createDataFrame(Seq(
//      (1, "This is my first sentence. This is my second."),
//      (2, "This is my third sentence. This is my forth.")
//    )).toDF("id", "text")
//
//    val expectedTokens = Seq(
//      Annotation(TOKEN, 0, 3, "this", Map("sentence" -> "0")),
//      Annotation(TOKEN, 5, 6, "is", Map("sentence" -> "0")),
//      Annotation(TOKEN, 8, 9, "my", Map("sentence" -> "0")),
//      Annotation(TOKEN, 11, 15, "first", Map("sentence" -> "0")),
//      Annotation(TOKEN, 17, 25, "sentence.", Map("sentence" -> "0")),
//      Annotation(TOKEN, 27, 30, "this", Map("sentence" -> "1")),
//      Annotation(TOKEN, 32, 33, "is", Map("sentence" -> "1")),
//      Annotation(TOKEN, 35, 36, "my", Map("sentence" -> "1")),
//      Annotation(TOKEN, 38, 44, "second.", Map("sentence" -> "1")),
//      Annotation(TOKEN, 0, 3, "this", Map("sentence" -> "0")),
//      Annotation(TOKEN, 5, 6, "is", Map("sentence" -> "0")),
//      Annotation(TOKEN, 8, 9, "my", Map("sentence" -> "0")),
//      Annotation(TOKEN, 11, 15, "third", Map("sentence" -> "0")),
//      Annotation(TOKEN, 17, 25, "sentence.", Map("sentence" -> "0")),
//      Annotation(TOKEN, 27, 30, "this", Map("sentence" -> "1")),
//      Annotation(TOKEN, 32, 33, "is", Map("sentence" -> "1")),
//      Annotation(TOKEN, 35, 36, "my", Map("sentence" -> "1")),
//      Annotation(TOKEN, 38, 43, "forth.", Map("sentence" -> "1"))
//    )
//
//    val documentAssembler = new DocumentAssembler()
//      .setInputCol("text")
//      .setOutputCol("document")
//
//    val sentence = new SentenceDetector()
//      .setInputCols("document")
//      .setOutputCol("sentence")
//
//    val regexTokenizer = new RegexTokenizer()
//      .setInputCols(Array("sentence"))
//      .setOutputCol("regexToken")
//      .setToLowercase(true)
//      .setPattern("\\s+")
//
//    val pipeline = new Pipeline()
//      .setStages(Array(
//        documentAssembler,
//        sentence,
//        regexTokenizer
//      ))
//
//    val pipelineDF = pipeline.fit(testData).transform(testData)
//
//    //    pipelineDF.select(size(pipelineDF("regexToken.result")).as("totalTokens")).show
//    //    pipelineDF.select(pipelineDF("document")).show(false)
//    //    pipelineDF.select(pipelineDF("sentence")).show(false)
//    //    pipelineDF.select(pipelineDF("regexToken.result")).show(false)
//    //    pipelineDF.select(pipelineDF("regexToken")).show(false)
//
//    val regexTokensResults = Annotation.collect(pipelineDF, "regexToken").flatten.toSeq
//    assert(regexTokensResults == expectedTokens)
//
//  }
//
//  "RegexTokenizer" should "correctly tokenize by patterns" in {
//
//    val testData = ResourceHelper.spark.createDataFrame(Seq(
//      (1, "T1-T2 DATE**[12/24/13] 10/12, ph+ 90%"))).toDF("id", "text")
//
//    val expectedTokens = Seq(
//      Annotation(TOKEN, 0, 1, "t1", Map("sentence" -> "0")),
//      Annotation(TOKEN, 3, 4, "t2", Map("sentence" -> "0")),
//      Annotation(TOKEN, 6, 9, "date", Map("sentence" -> "0")),
//      Annotation(TOKEN, 12, 21, "[12/24/13]", Map("sentence" -> "0")),
//      Annotation(TOKEN, 23, 27, "10/12", Map("sentence" -> "0")),
//      Annotation(TOKEN, 30, 32, "ph+", Map("sentence" -> "0")),
//      Annotation(TOKEN, 34, 36, "90%", Map("sentence" -> "0"))
//    )
//
//    val documentAssembler = new DocumentAssembler()
//      .setInputCol("text")
//      .setOutputCol("document")
//
//    val sentence = new SentenceDetector()
//      .setInputCols("document")
//      .setOutputCol("sentence")
//
//    val regexTokenizer = new RegexTokenizer()
//      .setInputCols(Array("sentence"))
//      .setOutputCol("regexToken")
//      .setToLowercase(true)
//      .setPattern("([^a-zA-Z\\/0-9\\[\\]+%])")
//
//    val pipeline = new Pipeline()
//      .setStages(Array(
//        documentAssembler,
//        sentence,
//        regexTokenizer
//      ))
//
//    val pipelineDF = pipeline.fit(testData).transform(testData)
//
//    //    pipelineDF.select(size(pipelineDF("regexToken.result")).as("totalTokens")).show
//    //    pipelineDF.select(pipelineDF("document")).show(false)
//    //    pipelineDF.select(pipelineDF("sentence")).show(false)
//    //    pipelineDF.select(pipelineDF("regexToken.result")).show(false)
//    //    pipelineDF.select(pipelineDF("regexToken")).show(false)
//
//    val regexTokensResults = Annotation.collect(pipelineDF, "regexToken").flatten.toSeq
//    assert(regexTokensResults == expectedTokens)
//
//  }

  "a Tokenizer" should "should correctly tokenize a parsed doc" in {
    val content = "1. T1-T2 DATE**[12/24/13] $1.99 () (10/12), ph+ 90%"
    println(s"content: $content")
    val pattern = "\\s+|(?=[-.:;*+,$&%\\[\\]])|(?<=[-.:;*+,$&%\\[\\]])"
    println(s"pattern: $pattern")

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
      .setOutputCol("token")
      .setPattern(pattern)

    //#.setSplitPattern("\s+|(?=[^a-zA-Z0-9_/])|(?<=[^a-zA-Z0-9_/])")
    //#.setSplitPattern("\s+|(?=[-.:;*+,$&%\[\]])|(?<=[-.:;*+,$&%\[\]])")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetect, tokenizer))

    val res = pipeline.fit(data).transform(data)

    res.select("token").collect().foreach {
      row => println(row.getSeq[Row](0).map(Annotation(_)).mkString("\n"))
    }
  }
}
