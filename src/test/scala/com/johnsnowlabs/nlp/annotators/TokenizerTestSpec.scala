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

  import SparkAccessor.spark.implicits._

  val regexTokenizer = new Tokenizer

  "a Tokenizer" should s"be of type ${AnnotatorType.TOKEN}" in {
    assert(regexTokenizer.annotatorType == AnnotatorType.TOKEN)
  }


  val targetText = "Hello, I won't be from New York in the U.S.A. (and you know it héroe). Give me my horse! or $100" +
    " bucks 'He said', I'll defeat markus-crassus. You understand. Goodbye George E. Bush. www.google.com."
  val expected = Array(
    "Hello", ",", "I", "wo", "n't", "be", "from", "New York", "in", "the", "U.S.A.", "(", "and", "you", "know", "it",
    "héroe", ")", ".", "Give", "me", "my", "horse", "!", "or", "$100", "bucks", "'", "He", "said", "'", ",", "I", "'ll",
    "defeat", "markus-crassus", ".", "You", "understand", ".", "Goodbye", "George", "E.", "Bush", ".", "www.google.com", "."
  )

  "a Tokenizer" should "correctly tokenize target text on its defaults parameters with composite" in {
    val data = DataBuilder.basicDataBuild(targetText)
    val document = new DocumentAssembler().setInputCol("text").setOutputCol("document")
    val tokenizer = new Tokenizer().setInputCols("document").setOutputCol("token").setCompositeTokensPatterns(Array("New York"))
    val finisher = new Finisher().setInputCols("token").setOutputAsArray(true).setCleanAnnotations(false).setOutputCols("output")
    val pipeline = new Pipeline().setStages(Array(document, tokenizer, finisher))
    val pip = pipeline.fit(data).transform(data)
    val result = pip
      .select("output").as[Array[String]]
      .collect.flatten
    assert(
      result.sameElements(expected),
      s"because result tokens differ: " +
        s"\nresult was \n${result.mkString("|")} \nexpected is: \n${expected.mkString("|")}"
    )
    pip
      .select("token").as[Array[Annotation]]
      .collect.foreach(annotations => {
      annotations.foreach(annotation => {
        assert(targetText.slice(annotation.begin, annotation.end + 1) == annotation.result)
      })
    })
  }

  "a Tokenizer" should "correctly tokenize target sentences on its defaults parameters with composite" in {
    val data = DataBuilder.basicDataBuild(targetText)
    val document = new DocumentAssembler().setInputCol("text").setOutputCol("document")
    val sentence = new SentenceDetector().setInputCols("document").setOutputCol("sentence")
    val tokenizer = new Tokenizer().setInputCols("sentence").setOutputCol("token").setCompositeTokensPatterns(Array("New York"))
    val finisher = new Finisher().setInputCols("token").setOutputAsArray(true).setOutputCols("output")
    val pipeline = new Pipeline().setStages(Array(document, sentence, tokenizer, finisher))
    val result = pipeline.fit(data).transform(data).select("output").as[Array[String]]
      .collect.flatten
    assert(
      result.sameElements(expected),
      s"because result tokens differ: " +
        s"\nresult was \n${result.mkString("|")} \nexpected is: \n${expected.mkString("|")}"
    )
  }

  "a Tokenizer" should "correctly tokenize target sentences on its defaults parameters with composite and different target pattern" in {
    val data = DataBuilder.basicDataBuild("Hello New York and Goodbye")
    val document = new DocumentAssembler().setInputCol("text").setOutputCol("document")
    val sentence = new SentenceDetector().setInputCols("document").setOutputCol("sentence")
    val tokenizer = new Tokenizer().setInputCols("sentence").setOutputCol("token").setTargetPattern("\\w+").setCompositeTokensPatterns(Array("New York"))
    val finisher = new Finisher().setInputCols("token").setOutputAsArray(true).setOutputCols("output")
    val pipeline = new Pipeline().setStages(Array(document, sentence, tokenizer, finisher))
    val result = pipeline.fit(data).transform(data).select("output").as[Array[String]]
      .collect.flatten
    assert(
      result.sameElements(Seq("Hello", "New York", "and", "Goodbye")),
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
      .setInputCols("document")
      .setOutputCol("token")
    val tokenized = tokenizer.transform(assembled)

    val date1 = new Date().getTime
    Annotation.take(tokenized, "token", 5000)
    info(s"Collected 5000 tokens took ${(new Date().getTime - date1) / 1000.0} seconds")
  }

  val latinBodyData: Dataset[Row] = DataBuilder.basicDataBuild(ContentProvider.latinBody)

  "A full Tokenizer pipeline with latin content" should behave like fullTokenizerPipeline(latinBodyData)

  "a tokenizer" should "handle composite tokens with special chars" in {

    val data = DataBuilder.basicDataBuild("Are you kidding me ?!?! what is this for !?")
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")

    val assembled = documentAssembler.transform(data)

    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")
      .setCompositeTokensPatterns(Array("Are you"))

    val tokenized = tokenizer.transform(assembled)
    val result = tokenized.collect()
  }

  "a silly tokenizer" should "split suffixes" in {

    val data = DataBuilder.basicDataBuild("One, after the\n\nOther, (and) again.\nPO, QAM,")
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("doc")
      .setTrimAndClearNewLines(false)

    val assembled = documentAssembler.transform(data)

    val tokenizer = new SimpleTokenizer()
      .setInputCols("doc")
      .setOutputCol("token")

    val tokenized = tokenizer.transform(assembled)
    val result = tokenized.select("token").as[Seq[Annotation]].collect.head.map(_.result)
    val expected = Seq("One", ",", "after", "the", "\n", "\n", "Other", ",", "(", "and", ")", "again", ".", "\n", "PO", ",", "QAM", ",")
    assert(result.equals(expected))

  }
}
