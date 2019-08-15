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

  val regexTokenizer = new Tokenizer()

  "a Tokenizer" should s"be of type ${AnnotatorType.TOKEN}" in {
    assert(regexTokenizer.outputAnnotatorType == AnnotatorType.TOKEN)
  }

  val ls = System.lineSeparator
  val lsl = ls.length

  def getTokenizerOutput[T](tokenizer: TokenizerModel, data: DataFrame, mode: String = "finisher"): Array[T] = {
    val finisher = new Finisher().setInputCols("token").setOutputAsArray(true).setCleanAnnotations(false).setOutputCols("output")
    val pipeline = new Pipeline().setStages(Array(tokenizer, finisher))
    val pip = pipeline.fit(data).transform(data)
    if (mode == "finisher") {
      pip.select("output").as[Array[String]].collect.flatten.asInstanceOf[Array[T]]
    } else {
      pip.select("token").as[Array[Annotation]].collect.flatten.asInstanceOf[Array[T]]
    }
  }


  val targetText1 = "Hello, I won't be from New York in the U.S.A. (and you know it héroe). Give me my horse! or $100" +
    " bucks 'He said', I'll defeat markus-crassus. You understand. Goodbye George E. Bush. www.google.com."
  val expected1 = Array(
    "Hello", ",", "I", "won't", "be", "from", "New York", "in", "the", "U.S.A", ".", "(", "and", "you", "know", "it",
    "héroe", ").", "Give", "me", "my", "horse", "!", "or", "$100", "bucks", "'", "He", "said", "',", "I'll",
    "defeat", "markus-crassus", ".", "You", "understand", ".", "Goodbye", "George", "E", ".", "Bush", ".", "www.google.com", "."
  )
  val expected1b = Array(
    "Hello", ",", "I", "won't", "be", "from", "New York", "in", "the", "U.S.A", ".", "(", "and", "you", "know", "it",
    "héroe", ").", "Give", "me", "my", "horse", "!", "or", "$100", "bucks", "'", "He", "said", "',", "I'll",
    "defeat", "markus", "-", "crassus", ".", "You", "understand", ".", "Goodbye", "George", "E", ".", "Bush", ".", "www.google.com", "."
  )

  "a Tokenizer" should "correctly tokenize target text on its defaults parameters with composite" in {

    val data = DataBuilder.basicDataBuild(targetText1)
    val tokenizer = new Tokenizer().setInputCols("document").setOutputCol("token")
      .setExceptions(Array("New York")).fit(data)
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

  "a Tokenizer" should "correctly tokenize target text on its defaults parameters with case insensitive composite" in {

    val data = DataBuilder.basicDataBuild(targetText1)
    val tokenizer = new Tokenizer().setInputCols("document").setOutputCol("token")
      .setCaseSensitiveExceptions(false)
      .setExceptions(Array("new york"))
      .fit(data)
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

  "a Tokenizer" should "correctly tokenize target sentences on its defaults parameters with composite" in {
    val data = DataBuilder.basicDataBuild(targetText1)
    val tokenizer = new Tokenizer().setInputCols("document").setOutputCol("token")
      .setExceptions(Array("New York"))
      .fit(data)
    val result = getTokenizerOutput[String](tokenizer, data)
    assert(
      result.sameElements(expected1),
      s"because result tokens differ: " +
        s"\nresult was \n${result.mkString("|")} \nexpected is: \n${expected1.mkString("|")}"
    )
  }

  "a Tokenizer" should "correctly tokenize target sentences with split chars" in {
    val data = DataBuilder.basicDataBuild(targetText1)
    val tokenizer = new Tokenizer().setInputCols("document").setOutputCol("token")
      .setExceptions(Array("New York"))
      .addSplitChars("-")
      .fit(data)
    val result = getTokenizerOutput[String](tokenizer, data)
    assert(
      result.sameElements(expected1b),
      s"because result tokens differ: " +
        s"\nresult was \n${result.mkString("|")} \nexpected is: \n${expected1b.mkString("|")}"
    )
  }


  val targetText2 = "I'd like to say we didn't expect that. Jane's boyfriend."
  val expected2 = Array(
    Annotation(AnnotatorType.TOKEN, 0, 2, "I'd", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 4, 7, "like", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 9, 10, "to", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 12, 14, "say", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 16, 17, "we", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 19, 24, "didn't", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 26, 31, "expect", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 33, 36, "that", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 37, 37, ".", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 39, 44, "Jane's", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 46, 54, "boyfriend", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 55, 55, ".", Map("sentence" -> "0"))
  )

  "a Tokenizer" should s"correctly tokenize a simple sentence on defaults" in {
    val data = DataBuilder.basicDataBuild(targetText2)
    val tokenizer = new Tokenizer().setInputCols("document").setOutputCol("token").fit(data)
    val result = getTokenizerOutput[Annotation](tokenizer, data, "annotation")
    assert(
      result.sameElements(expected2),
      s"because result tokens differ: " +
        s"\nresult was \n${result.mkString("|")} \nexpected is: \n${expected2.mkString("|")}"
    )
  }

  val targetText3 = s"I'd      like to say${ls}we didn't${ls+ls}" +
    s" expect\nthat. ${ls+ls} " +
    s"Jane's\\nboyfriend\tsaid.${ls+ls}"
  val expected3 = Array(
    Annotation(AnnotatorType.TOKEN, 0, 2, "I'd", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 4+5, 7+5, "like", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 9+5, 10+5, "to", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 12+5, 14+5, "say", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 15+5+lsl, 16+5+lsl, "we", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 18+5+lsl, 23+5+lsl, "didn't", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 25+5+(lsl*3), 30+5+(lsl*3), "expect", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 32+5+(lsl*3), 35+5+(lsl*3), "that", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 36+5+(lsl*3), 36+5+(lsl*3), ".", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 39+5+(lsl*5), 55+5+(lsl*5), "Jane's\\nboyfriend", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 57+5+(lsl*5), 60+5+(lsl*5), "said", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 61+5+(lsl*5), 61+5+(lsl*5), ".", Map("sentence" -> "0"))
  )

  "a Tokenizer" should s"correctly tokenize a sentence with breaking characters on defaults" in {
    val data = DataBuilder.basicDataBuild(targetText3)
    val tokenizer = new Tokenizer().setInputCols("document").setOutputCol("token").fit(data)
    val result = getTokenizerOutput[Annotation](tokenizer, data, "annotation")
    assert(
      result.sameElements(expected3),
      s"because result tokens differ: " +
        s"\nresult was \n${result.mkString("|")} \nexpected is: \n${expected3.mkString("|")}"
    )
  }

  val targetText4 = s"I'd      like to say${ls}we didn't${ls+ls}" +
    s" expect\nthat. ${ls+ls} " +
    s"Jane's\\nboyfriend\tsaid.${ls+ls}"
  val expected4 = Array(
    Annotation(AnnotatorType.TOKEN, 0, 2, "I'd", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 4, 7, "like", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 9, 10, "to", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 12, 14, "say", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 16, 17, "we", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 19, 24, "didn't", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 26, 31, "expect", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 33, 36, "that", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 37, 37, ".", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 39, 55, "Jane's\\nboyfriend", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 57, 60, "said", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 61, 61, ".", Map("sentence" -> "0"))
  )

  "a Tokenizer" should s"correctly tokenize a sentence with breaking characters on shrink cleanup" in {
    val data = DataBuilder.basicDataBuild(targetText4)(cleanupMode="shrink")
    val tokenizer = new Tokenizer().setInputCols("document").setOutputCol("token").fit(data)
    val result = getTokenizerOutput[Annotation](tokenizer, data, "annotation")
    assert(
      result.sameElements(expected4),
      s"because result tokens differ: " +
        s"\nresult was \n${result.mkString("|")} \nexpected is: \n${expected4.mkString("|")}"
    )
  }

  val targetText5 = s"I'd      like to say${ls}we didn't${ls+ls}" +
    s" expect\nthat. ${ls+ls} " +
    s"Jane's\\nboyfriend\tsaid.${ls+ls}"
  val expected5 = Array(
    Annotation(AnnotatorType.TOKEN, 0, 2, "I'd", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 4, 7, "like", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 9, 10, "to", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 12, 14, "say", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 16, 17, "we", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 19, 24, "didn't", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 26, 31, "expect", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 33, 36, "that", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 37, 37, ".", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 39, 44, "Jane's", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 46, 54, "boyfriend", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 56, 59, "said", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 60, 60, ".", Map("sentence" -> "0"))
  )

  "a tokenizer" should "split French apostrophe on left" in {

    val data = DataBuilder.basicDataBuild("l'une d'un l'un, des l'extrême des l'extreme")
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("doc")

    val assembled = documentAssembler.transform(data)

    val tokenizer = new Tokenizer()
      .setInputCols("doc")
      .setOutputCol("token")
      .setInfixPatterns(Array(
        "([\\p{L}\\w]+'{1})([\\p{L}\\w]+)"
      ))
      .fit(data)

    val tokenized = tokenizer.transform(assembled)
    val result = tokenized.select("token").as[Seq[Annotation]].collect.head.map(_.result)
    val expected = Seq("l'", "une", "d'", "un", "l'", "un", ",", "des", "l'", "extrême", "des", "l'", "extreme")
    assert(result.equals(expected),
      s"because result tokens differ: " +
        s"\nresult was \n${result.mkString("|")} \nexpected is: \n${expected.mkString("|")}")

  }

  "a Tokenizer" should s"correctly tokenize a sentence with breaking characters on shrink_full cleanup" in {
    val data = DataBuilder.basicDataBuild(targetText5)(cleanupMode="shrink_full")
    val tokenizer = new Tokenizer().setInputCols("document").setOutputCol("token").fit(data)
    val result = getTokenizerOutput[Annotation](tokenizer, data, "annotation")
    assert(
      result.sameElements(expected5),
      s"because result tokens differ: " +
        s"\nresult was \n${result.mkString("|")} \nexpected is: \n${expected5.mkString("|")}"
    )
  }

  val targetText6 = s"I'd      like to say${ls}we didn't${ls+ls}" +
    s" expect\nthat. ${ls+ls} " +
    s"Jane's\\nboyfriend\tsaid.${ls+ls}"
  val expected6 = Array(
    Annotation(AnnotatorType.TOKEN, 0, 2, "I'd", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 4, 7, "like", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 9, 10, "to", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 12, 14, "say", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 16, 17, "we", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 19, 24, "didn't", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 26, 31, "expect", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 33, 37, "that.", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 39, 54, "Jane's boyfriend", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 56, 59, "said", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 60, 60, ".", Map("sentence" -> "0"))
  )

  "a Tokenizer" should s"correctly tokenize cleanup with composite and exceptions" in {
    val data = DataBuilder.basicDataBuild(targetText6)(cleanupMode="shrink_full")
    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")
      .addException("Jane's \\w+")
      .addException("that.")
      .fit(data)
    val result = getTokenizerOutput[Annotation](tokenizer, data, "annotation")
    assert(
      result.sameElements(expected6),
      s"because result tokens differ: " +
        s"\nresult was \n${result.mkString("|")} \nexpected is: \n${expected6.mkString("|")}"
    )
  }

  "a Tokenizer" should "correctly tokenize target sentences on its defaults parameters with composite and different target pattern" in {
    val data = DataBuilder.basicDataBuild("Hello New York and Goodbye")
    val tokenizer = new Tokenizer().setInputCols("document").setOutputCol("token")
      .setTargetPattern("\\w+")
      .setExceptions(Array("New York"))
      .fit(data)
    val result = getTokenizerOutput[String](tokenizer, data)
    assert(
      result.sameElements(Seq("Hello", "New York", "and", "Goodbye")),
      s"because result tokens differ: " +
        s"\nresult was \n${result.mkString("|")} \nexpected is: \n${expected1.mkString("|")}"
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
      .fit(data)
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
      .setExceptions(Array("Are you"))
      .fit(data)

    val tokenized = tokenizer.transform(assembled)
    val result = tokenized.collect()
  }

  "a silly tokenizer" should "split suffixes" in {

    val data = DataBuilder.basicDataBuild("One, after the\n\nOther, (and) again.\nPO, QAM,")
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("doc")

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
