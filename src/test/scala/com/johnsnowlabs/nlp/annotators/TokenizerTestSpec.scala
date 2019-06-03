package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp._
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.scalatest._
import java.util.Date

import org.apache.spark.ml.Pipeline

/**
  * Created by saif on 02/05/17.
  */
class TokenizerTestSpec extends FlatSpec with TokenizerBehaviors {

  import SparkAccessor.spark.implicits._

  val regexTokenizer = new Tokenizer

  "a Tokenizer" should s"be of type ${AnnotatorType.TOKEN}" ignore {
    assert(regexTokenizer.outputAnnotatorType == AnnotatorType.TOKEN)
  }


  val targetText1 = "Hello, I won't be from New York in the U.S.A. (and you know it héroe). Give me my horse! or $100" +
    " bucks 'He said', I'll defeat markus-crassus. You understand. Goodbye George E. Bush. www.google.com."
  val expected1 = Array(
    "Hello", ",", "I", "wo", "n't", "be", "from", "New York", "in", "the", "U.S.A.", "(", "and", "you", "know", "it",
    "héroe", ")", ".", "Give", "me", "my", "horse", "!", "or", "$100", "bucks", "'", "He", "said", "'", ",", "I", "'ll",
    "defeat", "markus-crassus", ".", "You", "understand", ".", "Goodbye", "George", "E.", "Bush", ".", "www.google.com", "."
  )

  val targetText2 = "I'd like to say we didn't expect that. Jane's boyfriend."
  val expected2 = Array(
    Annotation(AnnotatorType.TOKEN, 0, 0, "I", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 1, 2, "'d", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 4, 7, "like", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 9, 10, "to", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 12, 14, "say", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 16, 17, "we", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 19, 21, "did", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 22, 24, "n't", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 26, 31, "expect", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 33, 36, "that", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 37, 37, ".", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 39, 42, "Jane", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 43, 44, "'s", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 46, 54, "boyfriend", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 55, 55, ".", Map("sentence" -> "0"))
  )

  val ls = System.lineSeparator
  val lsl = ls.length

  val targetText3 = s"I'd      like to say${ls}we didn't${ls+ls}" +
    s" expect that. ${ls+ls} " +
    s"Jane's\\nboyfriend.${ls+ls}"
  val expected3 = Array(
    Annotation(AnnotatorType.TOKEN, 0, 0, "I", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 1, 2, "'d", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 4+5, 7+5, "like", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 9+5, 10+5, "to", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 12+5, 14+5, "say", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 15+5+lsl, 16+5+lsl, "we", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 18+5+lsl, 20+5+lsl, "did", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 21+5+lsl, 23+5+lsl, "n't", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 25+5+(lsl*3), 30+5+(lsl*3), "expect", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 32+5+(lsl*3), 35+5+(lsl*3), "that", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 36+5+(lsl*3), 36+5+(lsl*3), ".", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 39+5+(lsl*5), 42+5+(lsl*5), "Jane", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 43+5+(lsl*5), 44+5+(lsl*5), "'s", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 46+5+(lsl*5), 54+5+(lsl*5), "boyfriend", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 56+5+(lsl*5), 56+5+(lsl*5), ".", Map("sentence" -> "0"))
  )

  def getTokenizerOutput[T](tokenizer: Tokenizer, data: DataFrame, mode: String = "finisher"): Array[T] = {
    val document = new DocumentAssembler().setInputCol("text").setOutputCol("document").setTrimAndClearNewLines(false)
    val finisher = new Finisher().setInputCols("token").setOutputAsArray(true).setCleanAnnotations(false).setOutputCols("output")
    val pipeline = new Pipeline().setStages(Array(document, tokenizer, finisher))
    val pip = pipeline.fit(data).transform(data)
    if (mode == "finisher") {
      pip.select("output").as[Array[String]].collect.flatten.asInstanceOf[Array[T]]
    } else {
      pip.select("token").as[Array[Annotation]].collect.flatten.asInstanceOf[Array[T]]
    }
  }

  "a Tokenizer" should "correctly tokenize target text on its defaults parameters with composite" ignore {

    val data = DataBuilder.basicDataBuild(targetText1)
    val tokenizer = new Tokenizer().setInputCols("document").setOutputCol("token").setCompositeTokensPatterns(Array("New York"))
    val result = getTokenizerOutput[String](tokenizer, data)
    assert(
      result.sameElements(expected1),
      s"because result tokens differ: " +
        s"\nresult was \n${result.mkString("|")} \nexpected is: \n${expected1.mkString("|")}"
    )
    val result2 = getTokenizerOutput[Annotation](tokenizer, data, "annotation")
    result2.foreach(annotation => {
      assert(targetText1.slice(annotation.begin, annotation.end + 1) == annotation.result)
    })
  }

  "a Tokenizer" should s"correctly label ${targetText2.take(10).mkString("")+"..."}" in {
    val data = DataBuilder.basicDataBuild(targetText2)
    val tokenizer = new Tokenizer().setInputCols("document").setOutputCol("token").setCompositeTokensPatterns(Array("New York"))
    val result = getTokenizerOutput[Annotation](tokenizer, data, "annotation")
    assert(
      result.sameElements(expected2),
      s"because result tokens differ: " +
        s"\nresult was \n${result.mkString("|")} \nexpected is: \n${expected2.mkString("|")}"
    )
  }

  "a Tokenizer" should s"correctly label ${targetText3.take(10).mkString("")+"..."}" in {
    val data = DataBuilder.basicDataBuild(targetText3)
    val tokenizer = new Tokenizer().setInputCols("document").setOutputCol("token").setCompositeTokensPatterns(Array("New York"))
    val result = getTokenizerOutput[Annotation](tokenizer, data, "annotation")
    assert(
      result.sameElements(expected3),
      s"because result tokens differ: " +
        s"\nresult was \n${result.mkString("|")} \nexpected is: \n${expected3.mkString("|")}"
    )
  }

  "a Tokenizer" should "correctly tokenize target sentences on its defaults parameters with composite" ignore {
    val data = DataBuilder.basicDataBuild(targetText1)
    val tokenizer = new Tokenizer().setInputCols("document").setOutputCol("token").setCompositeTokensPatterns(Array("New York"))
    val result = getTokenizerOutput[String](tokenizer, data)
    assert(
      result.sameElements(expected1),
      s"because result tokens differ: " +
        s"\nresult was \n${result.mkString("|")} \nexpected is: \n${expected1.mkString("|")}"
    )
  }

  "a Tokenizer" should "correctly tokenize target sentences on its defaults parameters with composite and different target pattern" ignore {
    val data = DataBuilder.basicDataBuild("Hello New York and Goodbye")
    val tokenizer = new Tokenizer().setInputCols("document").setOutputCol("token").setTargetPattern("\\w+").setCompositeTokensPatterns(Array("New York"))
    val result = getTokenizerOutput[String](tokenizer, data)
    assert(
      result.sameElements(Seq("Hello", "New York", "and", "Goodbye")),
      s"because result tokens differ: " +
        s"\nresult was \n${result.mkString("|")} \nexpected is: \n${expected1.mkString("|")}"
    )
  }

  "a spark based tokenizer" should "resolve big data" ignore {
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

  "a tokenizer" should "handle composite tokens with special chars" ignore {

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

  "a silly tokenizer" should "split suffixes" ignore {

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
