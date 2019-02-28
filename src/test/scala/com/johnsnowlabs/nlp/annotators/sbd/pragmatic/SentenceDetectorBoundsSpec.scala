package com.johnsnowlabs.nlp.annotators.sbd.pragmatic

import com.johnsnowlabs.nlp.{Annotation, AnnotatorType, ContentProvider, DocumentAssembler}
import com.johnsnowlabs.nlp.annotators.common.Sentence
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.scalatest.FlatSpec


class SentenceDetectorBoundsSpec extends FlatSpec {

  "SentenceDetector" should "return correct sentence bounds" in {
    val model = new DefaultPragmaticMethod(false)
    val text = "Hello World!! New Sentence"
    val bounds = model.extractBounds(text)

    assert(bounds.length == 2)
    assert(bounds(0) == Sentence("Hello World!!", 0, 12, 0))
    assert(bounds(1) == Sentence("New Sentence", 14, 25, 1))

    checkBounds(text, bounds)
  }

  "SentenceDetector" should "correct return sentence bounds with whitespaces" in {
    val model = new DefaultPragmaticMethod(false)
    val text = " Hello World!! .  New Sentence  "
    val bounds = model.extractBounds(text)

    assert(bounds.length == 3)
    assert(bounds(0) == Sentence("Hello World!!", 1, 13, 0))
    assert(bounds(1) == Sentence(".", 15, 15, 1))
    assert(bounds(2) == Sentence("New Sentence", 18, 29, 2))

    checkBounds(text, bounds)
  }

  "SentenceDetector" should "correct process custom delimiters" in {
    val model = new MixedPragmaticMethod(false, Array("\n\n"))
    val text = " Hello World.\n\nNew Sentence\n\nThird"
    val bounds = model.extractBounds(" Hello World.\n\nNew Sentence\n\nThird")

    assert(bounds.length == 3)
    assert(bounds(0) == Sentence("Hello World.", 1, 12, 0))
    assert(bounds(1) == Sentence("New Sentence", 15, 26, 1))
    assert(bounds(2) == Sentence("Third", 29, 33, 2))

    checkBounds(text, bounds)
  }

  "SentenceDetector" should "correct process custom delimiters in with dots" in {
    val model = new MixedPragmaticMethod(false, Array("\n\n"))
    val bounds = model.extractBounds(ContentProvider.conllEightSentences)

    assert(bounds.length == 8)
  }

  "SentenceDetector" should "successfully split long sentences" in {

    import ResourceHelper.spark.implicits._

    val sentence = "Hello world, this is a long sentence"

    val df = Seq(sentence).toDF("text")

    val expected = sentence.grouped(12).toArray

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sd = new SentenceDetector()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")
      .setMaxLength(12)

    val doc = document.transform(df)
    val sentenced = sd.transform(doc)
      .select("sentence")
      .as[Array[Annotation]].first

    assert(sentenced.length == expected.length)
    assert(sentenced.zip(expected).forall(r => r._1.result == r._2))
    assert(sentenced(0) == Annotation(AnnotatorType.DOCUMENT, 0, 11, "Hello world,", Map("sentence" -> "0")))
    assert(sentenced(1) == Annotation(AnnotatorType.DOCUMENT, 12, 23, " this is a l", Map("sentence" -> "1")))
    assert(sentenced(2) == Annotation(AnnotatorType.DOCUMENT, 24, 35, "ong sentence", Map("sentence" -> "2")))

  }


  private def checkBounds(text: String, bounds: Array[Sentence]) = {
    for (bound <- bounds) {
      assert(bound.content == text.substring(bound.start, bound.end + 1))
    }
  }
}
