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

package com.johnsnowlabs.nlp.annotators.sbd.pragmatic

import com.johnsnowlabs.nlp.annotators.common.Sentence
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType, ContentProvider, DocumentAssembler}
import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec

class SentenceDetectorBoundsSpec extends AnyFlatSpec {

  "SentenceDetector" should "support disable list detection" taggedAs FastTest in {
    val model = new DefaultPragmaticMethod(false, false)
    val text = "His age is 34. He is visiting hospital."
    val bounds = model.extractBounds(text)

    assert(bounds.length == 2)
    assert(bounds(0) == Sentence("His age is 34.", 0, 13, 0))
    assert(bounds(1) == Sentence("He is visiting hospital.", 15, 38, 1))

    checkBounds(text, bounds)
  }

  "SentenceDetector" should "return correct sentence bounds" taggedAs FastTest in {
    val model = new DefaultPragmaticMethod(false)
    val text = "Hello World!! New Sentence"
    val bounds = model.extractBounds(text)

    assert(bounds.length == 2)
    assert(bounds(0) == Sentence("Hello World!!", 0, 12, 0))
    assert(bounds(1) == Sentence("New Sentence", 14, 25, 1))

    checkBounds(text, bounds)
  }

  "SentenceDetector" should "correct return sentence bounds with whitespaces" taggedAs FastTest in {
    val model = new DefaultPragmaticMethod(false)
    val text = " Hello World!! .  New Sentence  "
    val bounds = model.extractBounds(text)

    assert(bounds.length == 3)
    assert(bounds(0) == Sentence("Hello World!!", 1, 13, 0))
    assert(bounds(1) == Sentence(".", 15, 15, 1))
    assert(bounds(2) == Sentence("New Sentence", 18, 29, 2))

    checkBounds(text, bounds)
  }

  "SentenceDetector" should "correct process custom delimiters" taggedAs FastTest in {
    val model = new MixedPragmaticMethod(false, true, Array("\n\n"))
    val text = " Hello World.\n\nNew Sentence\n\nThird"
    val bounds = model.extractBounds(" Hello World.\n\nNew Sentence\n\nThird")

    assert(bounds.length == 3)
    assert(bounds(0) == Sentence("Hello World.", 1, 12, 0))
    assert(bounds(1) == Sentence("New Sentence", 15, 26, 1))
    assert(bounds(2) == Sentence("Third", 29, 33, 2))

    checkBounds(text, bounds)
  }

  "SentenceDetector" should "correct process custom delimiters in with dots" taggedAs FastTest in {
    val model = new MixedPragmaticMethod(false, true, Array("\n\n"))
    val bounds = model.extractBounds(ContentProvider.conllEightSentences)

    assert(bounds.length == 8)
  }

  "SentenceDetector" should "successfully split long sentences" taggedAs FastTest in {

    import ResourceHelper.spark.implicits._

    val sentence = "Hello world, this is a long sentence longerThanSplitLength"

    val df = Seq(sentence).toDF("text")

    val expected = Array("Hello world,", "this is a", "long", "sentence", "longerThanSplitLength")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sd = new SentenceDetector()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")
      .setSplitLength(12)

    val doc = document.transform(df)
    val sentenced = sd
      .transform(doc)
      .select("sentence")
      .as[Array[Annotation]]
      .first

    assert(sentenced.length == expected.length)
    assert(
      sentenced
        .zip(expected)
        .forall(r => {
          println(r._1.result)
          println(r._2)
          r._1.result == r._2
        }))
    assert(
      sentenced(0) == Annotation(
        AnnotatorType.DOCUMENT,
        0,
        11,
        "Hello world,",
        Map("sentence" -> "0"),
        Array.emptyFloatArray))
    assert(
      sentenced(1) == Annotation(
        AnnotatorType.DOCUMENT,
        13,
        21,
        "this is a",
        Map("sentence" -> "1"),
        Array.emptyFloatArray))
    assert(
      sentenced(2) == Annotation(
        AnnotatorType.DOCUMENT,
        23,
        26,
        "long",
        Map("sentence" -> "2"),
        Array.emptyFloatArray))
    assert(
      sentenced(3) == Annotation(
        AnnotatorType.DOCUMENT,
        28,
        35,
        "sentence",
        Map("sentence" -> "3"),
        Array.emptyFloatArray))
    assert(
      sentenced(4) == Annotation(
        AnnotatorType.DOCUMENT,
        37,
        57,
        "longerThanSplitLength",
        Map("sentence" -> "4"),
        Array.emptyFloatArray))

  }

  "SentenceDetector" should "correctly filters out sentences less or greater than maxLength and minLength" taggedAs FastTest in {
    import ResourceHelper.spark.implicits._

    val sentence =
      "Small sentence. This is a normal sentence. This is a long sentence (longer than the maxLength)."

    val df = Seq(sentence).toDF("text")

    val expected = Array("This is a normal sentence.")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sd = new SentenceDetector()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")
      .setMinLength(16)
      .setMaxLength(26)

    val doc = document.transform(df)
    val sentenced = sd
      .transform(doc)
      .select("sentence")
      .as[Array[Annotation]]
      .first

    assert(sentenced.length == expected.length)
    assert(
      sentenced
        .zip(expected)
        .forall(r => {
          println(r._1.result)
          println(r._2)
          r._1.result == r._2
        }))
    assert(
      sentenced(0) == Annotation(
        AnnotatorType.DOCUMENT,
        16,
        41,
        "This is a normal sentence.",
        Map("sentence" -> "0"),
        Array.emptyFloatArray))

  }

  "SentenceDetector" should "support different strategies for returning custom bounds" taggedAs FastTest in {
    import ResourceHelper.spark.implicits._

    def assertSentenceBounds(
        sentences: String,
        sd: SentenceDetector,
        expected: Array[String]): Unit = {
      val df = Seq(sentences).toDF("text")

      val document = new DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")
      val doc = document.transform(df)

      val sentenced = sd
        .transform(doc)
        .select("sentence.result")
        .as[Array[String]]
        .first

      assert(sentenced.length == expected.length)
      sentenced.zip(expected).foreach { case (actual, expected) => assert(actual == expected) }
    }

    // Default Strategy: "none", replace all custom bounds
    val sdCustomOnly = new SentenceDetector()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")
      .setCustomBounds(Array("\\.", ";"))
      .setUseCustomBoundsOnly(true)

    val sentences =
      "This is a sentence. This one uses custom bounds; As is this one;"

    val expectedDefault =
      Array("This is a sentence", "This one uses custom bounds", "As is this one")
    assertSentenceBounds(sentences, sdCustomOnly, expectedDefault)

    val sdMixed = new SentenceDetector()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")
      .setCustomBounds(Array(";"))

    val expectedDefaultMixed =
      Array("This is a sentence.", "This one uses custom bounds", "As is this one")
    assertSentenceBounds(sentences, sdMixed, expectedDefaultMixed)

    // Strategy: "append", append sentence bound to match
    val sdCustomOnlyAppend = new SentenceDetector()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")
      .setCustomBounds(Array("\\.", ";"))
      .setUseCustomBoundsOnly(true)
      .setCustomBoundsStrategy("append")

    val expectedAppend =
      Array("This is a sentence.", "This one uses custom bounds;", "As is this one;")
    assertSentenceBounds(sentences, sdCustomOnlyAppend, expectedAppend)

    val sdMixedAppend = new SentenceDetector()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")
      .setCustomBounds(Array(";"))
      .setCustomBoundsStrategy("append")

    assertSentenceBounds(sentences, sdMixedAppend, expectedAppend)

    // Strategy: "prepend", append sentence bound to match
    val subHeaderList =
      """
        |1. This is a list
        |1.1 This is a subpoint
        |2. Second thing
        |2.2 Second subthing""".stripMargin

    val sdPrepend = new SentenceDetector()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")
      .setCustomBounds(Array(raw"\n[\d\.]+"))
      .setUseCustomBoundsOnly(true)
      .setCustomBoundsStrategy("prepend")

    val expectedPrepend = Array(
      "1. This is a list",
      "1.1 This is a subpoint",
      "2. Second thing",
      "2.2 Second subthing")
    assertSentenceBounds(subHeaderList, sdPrepend, expectedPrepend)

  }

  "SentenceDetector" should "throw an error if wrong custom bound strategy provided" taggedAs FastTest in {
    intercept[IllegalArgumentException] {
      new SentenceDetector()
        .setInputCols(Array("document"))
        .setOutputCol("sentence")
        .setCustomBounds(Array("\\.", ";"))
        .setUseCustomBoundsOnly(true)
        .setCustomBoundsStrategy("no thanks.")
    }
  }

  private def checkBounds(text: String, bounds: Array[Sentence]): Unit = {
    for (bound <- bounds) {
      assert(bound.content == text.substring(bound.start, bound.end + 1))
    }
  }
}
