/*
 * Copyright 2017-2022 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorType.TOKEN
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotator.SentenceDetector
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.{Pipeline, PipelineModel, Transformer}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.scalatest.flatspec.AnyFlatSpec

import java.util.Date
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

class TokenizerTestSpec extends AnyFlatSpec with TokenizerBehaviors {

  import SparkAccessor.spark.implicits._

  val tokenizer = new Tokenizer()

  val ls: String = System.lineSeparator
  val lsl: Int = ls.length

  val targetText0 =
    "My friend moved to New York. She likes it. Frank visited New York, and didn't like it."

  val targetText1: String =
    "Hello, I won't be from New York in the U.S.A. (and you know it héroe). Give me my horse! or $100" +
      " bucks 'He said', I'll defeat markus-crassus. You understand. Goodbye George E. Bush. www.google.com."
  val expected1: Array[String] = Array(
    "Hello",
    ",",
    "I",
    "won't",
    "be",
    "from",
    "New York",
    "in",
    "the",
    "U.S.A",
    ".",
    "(",
    "and",
    "you",
    "know",
    "it",
    "héroe",
    ").",
    "Give",
    "me",
    "my",
    "horse",
    "!",
    "or",
    "$100",
    "bucks",
    "'",
    "He",
    "said",
    "',",
    "I'll",
    "defeat",
    "markus-crassus",
    ".",
    "You",
    "understand",
    ".",
    "Goodbye",
    "George",
    "E",
    ".",
    "Bush",
    ".",
    "www.google.com",
    ".")
  val expected1b: Array[String] = Array(
    "Hello",
    ",",
    "I",
    "won't",
    "be",
    "from",
    "New York",
    "in",
    "the",
    "U.S.A",
    ".",
    "(",
    "and",
    "you",
    "know",
    "it",
    "héroe",
    ").",
    "Give",
    "me",
    "my",
    "horse",
    "!",
    "or",
    "$100",
    "bucks",
    "'",
    "He",
    "said",
    "',",
    "I'll",
    "defeat",
    "markus",
    "crassus",
    ".",
    "You",
    "understand",
    ".",
    "Goodbye",
    "George",
    "E",
    ".",
    "Bush",
    ".",
    "www.google.com",
    ".")

  val targetText2 = "I'd like to say we didn't expect that. Jane's boyfriend."

  val expected2: Array[Annotation] = Array(
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
    Annotation(AnnotatorType.TOKEN, 55, 55, ".", Map("sentence" -> "0")))

  val targetText3: String = s"I'd      like to say${ls}we didn't${ls + ls}" +
    s" expect\nthat. ${ls + ls} " +
    s"Jane's\\nboyfriend\tsaid.${ls + ls}"

  val expected3: Array[Annotation] = Array(
    Annotation(AnnotatorType.TOKEN, 0, 2, "I'd", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 4 + 5, 7 + 5, "like", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 9 + 5, 10 + 5, "to", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 12 + 5, 14 + 5, "say", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 15 + 5 + lsl, 16 + 5 + lsl, "we", Map("sentence" -> "0")),
    Annotation(AnnotatorType.TOKEN, 18 + 5 + lsl, 23 + 5 + lsl, "didn't", Map("sentence" -> "0")),
    Annotation(
      AnnotatorType.TOKEN,
      25 + 5 + (lsl * 3),
      30 + 5 + (lsl * 3),
      "expect",
      Map("sentence" -> "0")),
    Annotation(
      AnnotatorType.TOKEN,
      32 + 5 + (lsl * 3),
      35 + 5 + (lsl * 3),
      "that",
      Map("sentence" -> "0")),
    Annotation(
      AnnotatorType.TOKEN,
      36 + 5 + (lsl * 3),
      36 + 5 + (lsl * 3),
      ".",
      Map("sentence" -> "0")),
    Annotation(
      AnnotatorType.TOKEN,
      39 + 5 + (lsl * 5),
      55 + 5 + (lsl * 5),
      "Jane's\\nboyfriend",
      Map("sentence" -> "0")),
    Annotation(
      AnnotatorType.TOKEN,
      57 + 5 + (lsl * 5),
      60 + 5 + (lsl * 5),
      "said",
      Map("sentence" -> "0")),
    Annotation(
      AnnotatorType.TOKEN,
      61 + 5 + (lsl * 5),
      61 + 5 + (lsl * 5),
      ".",
      Map("sentence" -> "0")))

  val targetText4: String = s"I'd      like to say${ls}we didn't${ls + ls}" +
    s" expect\nthat. ${ls + ls} " +
    s"Jane's\\nboyfriend\tsaid.${ls + ls}"

  val expected4: Array[Annotation] = Array(
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
    Annotation(AnnotatorType.TOKEN, 61, 61, ".", Map("sentence" -> "0")))

  val targetText5: String = s"I'd      like to say${ls}we didn't${ls + ls}" +
    s" expect\nthat. ${ls + ls} " +
    s"Jane's\\nboyfriend\tsaid.${ls + ls}"

  val expected5: Array[Annotation] = Array(
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
    Annotation(AnnotatorType.TOKEN, 60, 60, ".", Map("sentence" -> "0")))

  val targetText6: String = s"I'd      like to say${ls}we didn't${ls + ls}" +
    s" expect\nthat. ${ls + ls} " +
    s"Jane's\\nboyfriend\tsaid.${ls + ls}"

  val expected6: Array[Annotation] = Array(
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
    Annotation(AnnotatorType.TOKEN, 60, 60, ".", Map("sentence" -> "0")))

  def getTokenizerOutput[T](
      tokenizer: TokenizerModel,
      data: DataFrame,
      mode: String = "finisher"): Array[T] = {
    val finisher = new Finisher()
      .setInputCols("token")
      .setOutputAsArray(true)
      .setCleanAnnotations(false)
      .setOutputCols("output")
    val pipeline = new Pipeline().setStages(Array(tokenizer, finisher))
    val pip = pipeline.fit(data).transform(data)
    if (mode == "finisher") {
      pip.select("output").as[Array[String]].collect.flatten.asInstanceOf[Array[T]]
    } else {
      pip.select("token").as[Array[Annotation]].collect.flatten.asInstanceOf[Array[T]]
    }
  }

  /* here assembler can either be a SentenceDetector or a DocumentAssembler */
  def getTokenizerPipelineOutput[T](
      assembler: Transformer,
      tokenizer: TokenizerModel,
      data: DataFrame,
      mode: String = "finisher"): Array[T] = {

    val finisher = new Finisher()
      .setInputCols("token")
      .setOutputAsArray(true)
      .setCleanAnnotations(false)
      .setOutputCols("output")
    val pipeline = new Pipeline().setStages(Array(assembler, tokenizer, finisher))
    val pip = pipeline.fit(data).transform(data)

    if (mode == "finisher") {
      pip.select("output").as[Array[String]].collect.flatten.asInstanceOf[Array[T]]
    } else {
      pip.select("token").as[Array[Annotation]].collect.flatten.asInstanceOf[Array[T]]
    }
  }

  private def assertIndexAlignment(tokens: Seq[Annotation], text: String): Unit = {
    tokens.foreach { case Annotation(_, start, end, result, _, _) =>
      val expected = text.substring(start, end + 1)
      assert(
        expected == result,
        s"Extracted Token \'$result\' did not match text($start, $end): \'$expected\'")
    }
  }

  behavior of "Tokenizer"

  it should s"be of type ${AnnotatorType.TOKEN}" taggedAs FastTest in {
    assert(tokenizer.outputAnnotatorType == AnnotatorType.TOKEN)
  }

  it should s"correctly handle exceptions in sentences and documents" taggedAs FastTest in {

    val data = DataBuilder.basicDataBuild(targetText0)

    val sentence = new SentenceDetector()
      .setInputCols(Array("document"))
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")
      .addException("New York")
      .fit(data)

    val result = getTokenizerPipelineOutput[Annotation](sentence, tokenizer, data, "annotation")
    assert(
      result(4) == Annotation(AnnotatorType.TOKEN, 19, 27, "New York.", Map("sentence" -> "0")))

    assert(
      result(11) == Annotation(AnnotatorType.TOKEN, 57, 65, "New York,", Map("sentence" -> "2")))

    assertIndexAlignment(result, targetText0)
  }

  it should s"correctly handle exceptionsPath" taggedAs FastTest in {

    val data = DataBuilder.basicDataBuild(targetText0)

    val sentence = new SentenceDetector()
      .setInputCols(Array("document"))
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")
      .setExceptionsPath("src/test/resources/token_exception_list.txt")
      .fit(data)

    val result = getTokenizerPipelineOutput[Annotation](sentence, tokenizer, data, "annotation")
    assert(
      result(4) == Annotation(AnnotatorType.TOKEN, 19, 27, "New York.", Map("sentence" -> "0")))

    assert(
      result(11) == Annotation(AnnotatorType.TOKEN, 57, 65, "New York,", Map("sentence" -> "2")))

    assertIndexAlignment(result, targetText0)
  }

  it should "correctly tokenize target text on its defaults parameters with composite" taggedAs FastTest in {

    val data = DataBuilder.basicDataBuild(targetText1)
    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")
      .setExceptions(Array("New York"))
      .fit(data)
    val result = getTokenizerOutput[String](tokenizer, data)
    assert(
      result.sameElements(expected1),
      s"because result tokens differ: " +
        s"\nresult was \n${result.mkString("|")} \nexpected is: \n${expected1.mkString("|")}")
    val result2 = getTokenizerOutput[Annotation](tokenizer, data, "annotation")
    assertIndexAlignment(result2, targetText1)
  }

  it should "correctly tokenize target text on its defaults parameters with case insensitive composite" taggedAs FastTest in {

    val data = DataBuilder.basicDataBuild(targetText1)
    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")
      .setCaseSensitiveExceptions(false)
      .setExceptions(Array("new york"))
      .fit(data)
    val result = getTokenizerOutput[String](tokenizer, data)
    assert(
      result.sameElements(expected1),
      s"because result tokens differ: " +
        s"\nresult was \n${result.mkString("|")} \nexpected is: \n${expected1.mkString("|")}")
    val result2 = getTokenizerOutput[Annotation](tokenizer, data, "annotation")
    assertIndexAlignment(result2, targetText1)

  }

  it should "correctly tokenize target sentences on its defaults parameters with composite" taggedAs FastTest in {
    val data = DataBuilder.basicDataBuild(targetText1)
    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")
      .setExceptions(Array("New York"))
      .fit(data)
    val result = getTokenizerOutput[String](tokenizer, data)
    val resultAnnotation = getTokenizerOutput[Annotation](tokenizer, data, "")
    assert(
      result.sameElements(expected1),
      s"because result tokens differ: " +
        s"\nresult was \n${result.mkString("|")} \nexpected is: \n${expected1.mkString("|")}")
    assertIndexAlignment(resultAnnotation, targetText1)
  }

  it should "correctly tokenize target sentences with split chars" taggedAs FastTest in {
    val data = DataBuilder.basicDataBuild(targetText1)
    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")
      .setExceptions(Array("New York"))
      .addSplitChars("-")
      .fit(data)
    val result = getTokenizerOutput[String](tokenizer, data)
    val resultAnnotation = getTokenizerOutput[Annotation](tokenizer, data, "")

    assert(
      result.sameElements(expected1b),
      s"because result tokens differ: " +
        s"\nresult was \n${result.mkString("|")} \nexpected is: \n${expected1b.mkString("|")}")
    assertIndexAlignment(resultAnnotation, targetText1)
  }

  it should s"correctly tokenize a simple sentence on defaults" taggedAs FastTest in {
    val data = DataBuilder.basicDataBuild(targetText2)
    val tokenizer = new Tokenizer().setInputCols("document").setOutputCol("token").fit(data)
    val result = getTokenizerOutput[Annotation](tokenizer, data, "annotation")
    assert(
      result.sameElements(expected2),
      s"because result tokens differ: " +
        s"\nresult was \n${result.mkString("|")} \nexpected is: \n${expected2.mkString("|")}")
    assertIndexAlignment(result, targetText2)
  }

  it should s"correctly tokenize a sentence with breaking characters on defaults" taggedAs FastTest in {
    val data = DataBuilder.basicDataBuild(targetText3)
    val tokenizer = new Tokenizer().setInputCols("document").setOutputCol("token").fit(data)
    val result = getTokenizerOutput[Annotation](tokenizer, data, "annotation")
    assert(
      result.sameElements(expected3),
      s"because result tokens differ: " +
        s"\nresult was \n${result.mkString("|")} \nexpected is: \n${expected3.mkString("|")}")
    assertIndexAlignment(result, targetText3)
  }

  it should s"correctly tokenize a sentence with breaking characters on shrink cleanup" taggedAs FastTest in {
    val data = DataBuilder.basicDataBuild(targetText4)(cleanupMode = "shrink")
    val tokenizer = new Tokenizer().setInputCols("document").setOutputCol("token").fit(data)
    val result = getTokenizerOutput[Annotation](tokenizer, data, "annotation")
    assert(
      result.sameElements(expected4),
      s"because result tokens differ: " +
        s"\nresult was \n${result.mkString("|")} \nexpected is: \n${expected4.mkString("|")}")

    val cleanedText = Annotation.collect(data, "document").head.head.result
    assertIndexAlignment(result, cleanedText)
  }

  it should "split French apostrophe on left" taggedAs FastTest in {

    val data = DataBuilder.basicDataBuild("l'une d'un l'un, des l'extrême des l'extreme")
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("doc")

    val assembled = documentAssembler.transform(data)

    val tokenizer = new Tokenizer()
      .setInputCols("doc")
      .setOutputCol("token")
      .setInfixPatterns(Array("([\\p{L}\\w]+'{1})([\\p{L}\\w]+)"))
      .fit(data)

    val tokenized = tokenizer.transform(assembled)
    val result = tokenized.select("token").as[Seq[Annotation]].collect.head.map(_.result)
    val expected = Seq(
      "l'",
      "une",
      "d'",
      "un",
      "l'",
      "un",
      ",",
      "des",
      "l'",
      "extrême",
      "des",
      "l'",
      "extreme")
    assert(
      result.equals(expected),
      s"because result tokens differ: " +
        s"\nresult was \n${result.mkString("|")} \nexpected is: \n${expected.mkString("|")}")

  }

  it should s"correctly tokenize a sentence with breaking characters on shrink_full cleanup" taggedAs FastTest in {
    val data = DataBuilder.basicDataBuild(targetText5)(cleanupMode = "shrink_full")
    val tokenizer = new Tokenizer().setInputCols("document").setOutputCol("token").fit(data)
    val result = getTokenizerOutput[Annotation](tokenizer, data, "annotation")
    assert(
      result.sameElements(expected5),
      s"because result tokens differ: " +
        s"\nresult was \n${result.mkString("|")} \nexpected is: \n${expected5.mkString("|")}")
  }

  it should s"correctly tokenize cleanup with composite and exceptions" taggedAs FastTest in {
    val data = DataBuilder.basicDataBuild(targetText6)(cleanupMode = "shrink_full")
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
        s"\nresult was \n${result.mkString("|")} \nexpected is: \n${expected6.mkString("|")}")

    val cleanedText = Annotation.collect(data, "document").head.head.result
    assertIndexAlignment(result, cleanedText)
  }

  it should "correctly tokenize target sentences on its defaults parameters with composite and different target pattern" taggedAs FastTest in {
    val text = "Hello New York and Goodbye"
    val data = DataBuilder.basicDataBuild(text)
    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")
      .setTargetPattern("\\w+")
      .setExceptions(Array("New York"))
      .fit(data)
    val result = getTokenizerOutput[String](tokenizer, data)
    val resultAnnotation = getTokenizerOutput[Annotation](tokenizer, data, "")
    assert(
      result.sameElements(Seq("Hello", "New York", "and", "Goodbye")),
      s"because result tokens differ: " +
        s"\nresult was \n${result.mkString("|")} \nexpected is: \n${expected1.mkString("|")}")
    assertIndexAlignment(resultAnnotation, text)
  }

  it should "resolve big data" taggedAs FastTest in {
    val data = ContentProvider.parquetData
      .limit(500000)
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

  it should "handle composite tokens with special chars" taggedAs FastTest in {

    val text = "Are you kidding me ?!?! what is this for !?"
    val data = DataBuilder.basicDataBuild(text)
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")

    val assembled = documentAssembler.transform(data)

    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")
      .setExceptions(Array("Are you"))
      .fit(data)

    val tokenized = tokenizer.transform(assembled)
    val resultAnnotation = Annotation.collect(tokenized, "token").flatten

    assertIndexAlignment(resultAnnotation, text)
  }

  it should "correctly filter out tokens based on setting minimum and maximum lengths" taggedAs FastTest in {
    val data = DataBuilder.basicDataBuild("Hello New York and Goodbye")
    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")
      .setMinLength(4)
      .setMaxLength(5)
      .fit(data)

    val result = getTokenizerOutput[String](tokenizer, data)
    assert(
      result.sameElements(Seq("Hello", "York")),
      s"because result tokens differ: " +
        s"\nresult was \n${result.mkString("|")} \nexpected is: \n${expected1.mkString("|")}")
  }

  it should "work correctly with multiple split chars" taggedAs FastTest in {
    val text = "Hello big-city-of-lights welcome to the ground###earth."
    val data =
      DataBuilder.basicDataBuild(text)
    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")
      .setSplitChars(Array("-", "#"))
      .fit(data)

    val expected = Seq(
      Annotation("token", 0, 4, "Hello", Map("sentence" -> "0")),
      Annotation("token", 6, 8, "big", Map("sentence" -> "0")),
      Annotation("token", 10, 13, "city", Map("sentence" -> "0")),
      Annotation("token", 15, 16, "of", Map("sentence" -> "0")),
      Annotation("token", 18, 23, "lights", Map("sentence" -> "0")),
      Annotation("token", 25, 31, "welcome", Map("sentence" -> "0")),
      Annotation("token", 33, 34, "to", Map("sentence" -> "0")),
      Annotation("token", 36, 38, "the", Map("sentence" -> "0")),
      Annotation("token", 40, 45, "ground", Map("sentence" -> "0")),
      Annotation("token", 49, 53, "earth", Map("sentence" -> "0")),
      Annotation("token", 54, 54, ".", Map("sentence" -> "0")))
    val result = getTokenizerOutput[Annotation](tokenizer, data, "annotation")
    assert(
      result.sameElements(expected),
      s"because result tokens differ: " +
        s"\nresult was \n${result.mkString("\n")} \nexpected is: \n${expected.mkString("\n")}")

    assertIndexAlignment(result, text)
  }

  it should "work correctly with multiple split chars including stars '*'" taggedAs FastTest in {
    val text = "Hello big-city-of-lights welcome to*the ground###earth."
    val data =
      DataBuilder.basicDataBuild(text)
    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")
      .setSplitChars(Array("-", "#", "\\*"))
      .fit(data)

    val expected = Seq(
      Annotation("token", 0, 4, "Hello", Map("sentence" -> "0")),
      Annotation("token", 6, 8, "big", Map("sentence" -> "0")),
      Annotation("token", 10, 13, "city", Map("sentence" -> "0")),
      Annotation("token", 15, 16, "of", Map("sentence" -> "0")),
      Annotation("token", 18, 23, "lights", Map("sentence" -> "0")),
      Annotation("token", 25, 31, "welcome", Map("sentence" -> "0")),
      Annotation("token", 33, 34, "to", Map("sentence" -> "0")),
      Annotation("token", 36, 38, "the", Map("sentence" -> "0")),
      Annotation("token", 40, 45, "ground", Map("sentence" -> "0")),
      Annotation("token", 49, 53, "earth", Map("sentence" -> "0")),
      Annotation("token", 54, 54, ".", Map("sentence" -> "0")))
    val result = getTokenizerOutput[Annotation](tokenizer, data, "annotation")
    assert(
      result.sameElements(expected),
      s"because result tokens differ: " +
        s"\nresult was \n${result.mkString("\n")} \nexpected is: \n${expected.mkString("\n")}")

    assertIndexAlignment(result, text)
  }

  it should "work correctly with a split pattern" taggedAs FastTest in {
    val text = "Hello big-city-of-lights welcome to the ground###earth."
    val data =
      DataBuilder.basicDataBuild(text)
    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")
      .setSplitPattern("-|#")
      .fit(data)
    val expected = Seq(
      Annotation("token", 0, 4, "Hello", Map("sentence" -> "0")),
      Annotation("token", 6, 8, "big", Map("sentence" -> "0")),
      Annotation("token", 10, 13, "city", Map("sentence" -> "0")),
      Annotation("token", 15, 16, "of", Map("sentence" -> "0")),
      Annotation("token", 18, 23, "lights", Map("sentence" -> "0")),
      Annotation("token", 25, 31, "welcome", Map("sentence" -> "0")),
      Annotation("token", 33, 34, "to", Map("sentence" -> "0")),
      Annotation("token", 36, 38, "the", Map("sentence" -> "0")),
      Annotation("token", 40, 45, "ground", Map("sentence" -> "0")),
      Annotation("token", 49, 53, "earth", Map("sentence" -> "0")),
      Annotation("token", 54, 54, ".", Map("sentence" -> "0")))
    val result = getTokenizerOutput[Annotation](tokenizer, data, "annotation")
    assert(
      result.sameElements(expected),
      s"because result tokens differ: " +
        s"\nresult was \n${result.mkString("\n")} \nexpected is: \n${expected.mkString("\n")}")

    assertIndexAlignment(result, text)
  }

  it should "benchmark exceptions" taggedAs SlowTest ignore {
    val data = AnnotatorBuilder.getTrainingDataSet("src/test/resources/spell/sherlockholmes.txt")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer))

    /** Run first to cache for more consistent results */
    val result: Array[Row] = pipeline.fit(data).transform(data).select("token.result").collect()

    val tokens = result
      .foldLeft(ArrayBuffer.empty[String]) { (arr: ArrayBuffer[String], i: Row) =>
        val Row(tokens: mutable.WrappedArray[String] @unchecked) = i
        arr ++= tokens.map(_.replaceAll("\\W", "")).filter(_.nonEmpty)
      }
      .toArray

    val documentAssembler2 = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizerWithExceptions = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")
      .setExceptions(tokens.slice(0, 200))

    val pipelineWithExceptions =
      new Pipeline().setStages(Array(documentAssembler2, tokenizerWithExceptions))

    Benchmark.measure(
      iterations = 20,
      forcePrint = true,
      description = "Time to tokenize Sherlock Holmes with exceptions") {
      pipelineWithExceptions.fit(data).transform(data).select("token.result").collect()
    }

  }

  it should "produce correct indexes for split patterns without space" taggedAs FastTest in {

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")
      .setSplitPattern("""\s+|(?=[-.:;*+,\(\)\/$&%\\[\\]])|(?<=[-.:;*+,\(\)\/$&%\\[\\]])""")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer))

    val text =
      "PRIMARY OBJECTIVES:\r\n\r\n      I. Determine the safety and feasibility of the chimeric antigen receptor T cells transduced\r\n      with the anti-CD19 (cluster of differentiation antigen 19 ) vector (referred to as CART-19\r\n      cells).\r\n\r\n      II. Determine duration of in vivo survival of CART-19 cells. RT-PCR (reverse transcription\r\n      polymerase chain reaction) analysis of whole blood will be used to detect and quantify\r\n      survival of CART-19 TCR (T-cell receptor) zeta:CD137 and TCR zeta cells over time.\r\n\r\n      SECONDARY OBJECTIVES:\r\n\r\n      I. For patients with detectable disease, measure anti-tumor response due to CART-19 cell\r\n      infusions.\r\n\r\n      II. To determine if the CD137 transgene is superior to the TCR zeta only transgene as\r\n      measured by the relative engraftment levels of CART-19 TCR zeta:CD137 and TCR zeta cells over\r\n      time.\r\n\r\n      III. Estimate relative trafficking of CART-19 cells to tumor in bone marrow and lymph nodes.\r\n\r\n      IV. For patients with stored or accessible tumor cells (such as patients with active chronic\r\n      lymphocytic leukemia(CLL), acute lymphocytic leukemia (ALL), etc) determine tumor cell\r\n      killing by CART-19 cells in vitro.\r\n\r\n      V. Determine if cellular or humoral host immunity develops against the murine anti-CD19, and\r\n      assess correlation with loss of detectable CART-19 (loss of engraftment).\r\n\r\n      VI. Determine the relative subsets of CART-19 T cells (Tcm, Tem, and Treg).\r\n\r\n      OUTLINE: Patients are assigned to 1 group according to order of enrollment.\r\n\r\n      Patients receive anti-CD19-CAR (coupled with CD137 and CD3 zeta signalling\r\n      domains)vector-transduced autologous T cells on days 0,1, and 2 in the absence of disease\r\n      progression or unacceptable toxicity.\r\n\r\n      After completion of study treatment, patients are followed intensively for 6 months, every 3\r\n      months for 2 years, and annually thereafter for 13 years."

    val data = Seq(text).toDF("text")

    val result = pipeline.fit(data).transform(data)

    val collected =
      Annotation.collect(result, "token").flatten.toSeq

    assertIndexAlignment(collected, text)
  }

  it should "produce correct indexes for matches without space" in {
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")
      .setSplitPattern("(?=.)") // split at every character

    val text = "abcdef"
    val data = Seq(text).toDF("text")

    val expected: Array[Annotation] =
      text.toArray.zipWithIndex.map { case (t: Char, i: Int) =>
        Annotation("token", begin = i, end = i, t.toString, Map("sentence" -> "0"))
      }

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer))
    val result = pipeline.fit(data).transform(data)

    val resultTokens: Array[Annotation] =
      Annotation.collect(result, "token").flatten

    assert(expected sameElements resultTokens)
    assertIndexAlignment(resultTokens, text)

  }

  val latinBodyData: Dataset[Row] = DataBuilder.basicDataBuild(ContentProvider.latinBody)

  "A full Tokenizer pipeline with latin content" should behave like fullTokenizerPipeline(
    latinBodyData)

  "RecursiveTokenizer" should "split suffixes" taggedAs FastTest in {
    val text = "One, after the\n\nOther, (and) again.\nPO, QAM,"
    val data = DataBuilder.basicDataBuild(text)
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("doc")

    val assembled = documentAssembler.transform(data)

    val tokenizer = new RecursiveTokenizer()
      .setInputCols("doc")
      .setOutputCol("token")
      .fit(data)

    val tokenized = tokenizer.transform(assembled)
    val result = tokenized.select("token").as[Seq[Annotation]].collect.head.map(_.result)
    val resultAnnotation = Annotation.collect(tokenized, "token").flatten
    val expected = Seq(
      "One",
      ",",
      "after",
      "the",
      "\n",
      "\n",
      "Other",
      ",",
      "(",
      "and",
      ")",
      "again",
      ".",
      "\n",
      "PO",
      ",",
      "QAM",
      ",")
    assert(result.equals(expected))
    assertIndexAlignment(resultAnnotation, text)
  }

  "RecursiveTokenizer" should "be serializable as model" taggedAs FastTest in {

    val data = DataBuilder.basicDataBuild("One, after the\n\nOther, (and) again.\nPO, QAM,")
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("doc")

    val tokenizer = new RecursiveTokenizer()
      .setInputCols("doc")
      .setOutputCol("token")
      .setInfixes(Array("\n", "(", ")"))
      .setSuffixes(Array(".", ":", "%", ",", ";", "?", "'", "\"", ")", "]", "\n", "!", "'s"))
      .setWhitelist(
        Array(
          "it's",
          "that's",
          "there's",
          "he's",
          "she's",
          "what's",
          "let's",
          "who's",
          "It's",
          "That's",
          "There's",
          "He's",
          "She's",
          "What's",
          "Let's",
          "Who's"))
      .setPrefixes(Array("'", "\"", "(", "[", "\n"))

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer))

    val pipelineModel = pipeline.fit(data)
    pipelineModel.transform(data).show()

    pipelineModel.write.overwrite().save("./tmp_pipelineModel_with_recTokenizer")

    val loadedPipelineModel = PipelineModel.load("./tmp_pipelineModel_with_recTokenizer")
    loadedPipelineModel.transform(data).show()

  }

  it should "tokenize with sentence index" taggedAs FastTest in {

    val text = "This is a sentence. This is another one"
    val testDataSet = Seq(text).toDS.toDF("text")

    val expectedEntitiesFromText1: Array[Seq[Annotation]] = Array(
      Seq(
        Annotation(TOKEN, 0, 3, "This", Map("sentence" -> "0")),
        Annotation(TOKEN, 5, 6, "is", Map("sentence" -> "0")),
        Annotation(TOKEN, 8, 8, "a", Map("sentence" -> "0")),
        Annotation(TOKEN, 10, 17, "sentence", Map("sentence" -> "0")),
        Annotation(TOKEN, 18, 18, ".", Map("sentence" -> "0"))),
      Seq(
        Annotation(TOKEN, 20, 23, "This", Map("sentence" -> "1")),
        Annotation(TOKEN, 25, 26, "is", Map("sentence" -> "1")),
        Annotation(TOKEN, 28, 34, "another", Map("sentence" -> "1")),
        Annotation(TOKEN, 36, 38, "one", Map("sentence" -> "1"))))

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetector = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")
      .setExplodeSentences(true)

    val tokenizer = new Tokenizer()
      .setInputCols("sentence")
      .setOutputCol("token")

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, sentenceDetector, tokenizer))

    val resultDataSet = pipeline.fit(testDataSet).transform(testDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "token")

    AssertAnnotations.assertFields(expectedEntitiesFromText1, actualEntities)
  }

}
