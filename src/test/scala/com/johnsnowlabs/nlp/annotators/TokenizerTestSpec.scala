package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp._
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._
import java.util.Date

import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import org.apache.spark.ml.Pipeline

/**
  * Created by saif on 02/05/17.
  */
class TokenizerTestSpec extends FlatSpec with TokenizerBehaviors {

  val regexTokenizer = new Tokenizer

  "a Tokenizer" should s"be of type ${AnnotatorType.TOKEN}" in {
    assert(regexTokenizer.annotatorType == AnnotatorType.TOKEN)
  }

  "a Tokenizer" should "correctly tokenize target text on its defaults parameters" in {
    val data = DataBuilder.basicDataBuild("Hello, I am from the U.S.A. (and you know it). Give me my horse! 'He said', I'll defeat markus-crassus.")
    import data.sparkSession.implicits._
    val document = new DocumentAssembler().setInputCol("text").setOutputCol("document")
    val tokenizer = new Tokenizer().setInputCols("document").setOutputCol("token")
    val finisher = new Finisher().setInputCols("token").setOutputAsArray(true).setOutputCols("output")
    val pipeline = new Pipeline().setStages(Array(document, tokenizer, finisher))
    val expected = Array(
      "Hello", ",", "I", "am", "from", "the", "U.S.A.", "(", "and", "you", "know", "it", ")", ".",
      "Give", "me", "my", "horse", "!", "'", "He", "said", "'", ",", "I", "'", "ll", "defeat", "markus-crassus", "."
    )
    val result = pipeline.fit(data).transform(data).select("output").as[Array[String]]
      .collect.flatten
    assert(
      result.sameElements(expected),
      s"because result tokens differ: " +
        s"\nresult was \n${result.mkString("|")} \nexpected is: \n${expected.mkString("|")}"
    )
  }

  "a Tokenizer" should "correctly tokenize target sentences on its defaults parameters" in {
    val data = DataBuilder.basicDataBuild("Hello, I am from the U.S.A. (and you know it). Give me my horse! 'He said', I'll defeat markus-crassus.")
    import data.sparkSession.implicits._
    val document = new DocumentAssembler().setInputCol("text").setOutputCol("document")
    val sentence = new SentenceDetector().setInputCols("document").setOutputCol("sentence")
    val tokenizer = new Tokenizer().setInputCols("sentence").setOutputCol("token")
    val finisher = new Finisher().setInputCols("token").setOutputAsArray(true).setOutputCols("output")
    val pipeline = new Pipeline().setStages(Array(document, sentence, tokenizer, finisher))
    val expected = Array(
      "Hello", ",", "I", "am", "from", "the", "U.S.A.", "(", "and", "you", "know", "it", ")", ".",
      "Give", "me", "my", "horse", "!", "'", "He", "said", "'", ",", "I", "'", "ll", "defeat", "markus-crassus", "."
    )
    val result = pipeline.fit(data).transform(data).select("output").as[Array[String]]
      .collect.flatten
    assert(
      result.sameElements(expected),
      s"because result tokens differ: " +
        s"\nresult was \n${result.mkString("|")} \nexpected is: \n${expected.mkString("|")}"
    )
  }

  "a spark based tokenizer" should "resolve big data" in {
    val data = ContentProvider.parquetData.limit(500000)
      .repartition(16)

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")

    val assembled = documentAssembler.transform(data)

    val tokenizer = new Tokenizer()
      .setOutputCol("token")
    val tokenized = tokenizer.transform(assembled)

    val date1 = new Date().getTime
    Annotation.take(tokenized, "token", 5000)
    info(s"Collected 5000 tokens took ${(new Date().getTime - date1) / 1000} seconds")
  }

  val latinBodyData: Dataset[Row] = DataBuilder.basicDataBuild(ContentProvider.latinBody)

  "A full Tokenizer pipeline with latin content" should behave like fullTokenizerPipeline(latinBodyData)

}
