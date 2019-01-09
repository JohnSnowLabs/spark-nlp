package com.johnsnowlabs.nlp.annotators.sbd.pragmatic

import com.johnsnowlabs.nlp.ContentProvider
import com.johnsnowlabs.nlp.annotators.common.Sentence
import org.scalatest.FlatSpec


class SentenceDetectorBoundsSpec extends FlatSpec {

  "SentenceDetector" should "return correct sentence bounds" in {
    val model = new DefaultPragmaticMethod(false)
    val text = "Hello World!! New Sentence"
    val bounds = model.extractBounds(text)

    assert(bounds.length == 2)
    assert(bounds(0) == Sentence("Hello World!!", 0, 12))
    assert(bounds(1) == Sentence("New Sentence", 14, 25))

    checkBounds(text, bounds)
  }

  "SentenceDetector" should "correct return sentence bounds with whitespaces" in {
    val model = new DefaultPragmaticMethod(false)
    val text = " Hello World!! .  New Sentence  "
    val bounds = model.extractBounds(text)

    assert(bounds.length == 3)
    assert(bounds(0) == Sentence("Hello World!!", 1, 13))
    assert(bounds(1) == Sentence(".", 15, 15))
    assert(bounds(2) == Sentence("New Sentence", 18, 29))

    checkBounds(text, bounds)
  }

  "SentenceDetector" should "correct process custom delimiters" in {
    val model = new MixedPragmaticMethod(false, Array("\n\n"))
    val text = " Hello World.\n\nNew Sentence\n\nThird"
    val bounds = model.extractBounds(" Hello World.\n\nNew Sentence\n\nThird")

    assert(bounds.length == 3)
    assert(bounds(0) == Sentence("Hello World.", 1, 12))
    assert(bounds(1) == Sentence("New Sentence", 15, 26))
    assert(bounds(2) == Sentence("Third", 29, 33))

    checkBounds(text, bounds)
  }

  "SentenceDetector" should "correct process custom delimiters in with dots" in {
    val model = new MixedPragmaticMethod(false, Array("\n\n"))
    val bounds = model.extractBounds(ContentProvider.conllEightSentences)

    assert(bounds.length == 8)
  }


  private def checkBounds(text: String, bounds: Array[Sentence]) = {
    for (bound <- bounds) {
      assert(bound.content == text.substring(bound.start, bound.end + 1))
    }
  }
}
