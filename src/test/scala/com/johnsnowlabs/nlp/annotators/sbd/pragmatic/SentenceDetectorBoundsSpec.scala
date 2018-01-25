package com.johnsnowlabs.nlp.annotators.sbd.pragmatic

import com.johnsnowlabs.nlp.annotators.common.Sentence
import org.scalatest.FlatSpec


class SentenceDetectorBoundsSpec extends FlatSpec {

  val model = new PragmaticMethod(false)

  "SentenceDetectorModel" should "return correct sentence bounds" in {
    val bounds = model.extractBounds("Hello World!! New Sentence", Array.empty[String])

    assert(bounds.length == 2)
    assert(bounds(0) == Sentence("Hello World!!", 0, 12))
    assert(bounds(1) == Sentence("New Sentence", 14, 25))
  }

  "SentenceDetectorModel" should "correct return sentence bounds with whitespaces" in {
    val bounds = model.extractBounds(" Hello World!! .  New Sentence  ", Array.empty[String])

    assert(bounds.length == 3)
    assert(bounds(0) == Sentence("Hello World!!", 1, 13))
    assert(bounds(1) == Sentence(".", 15, 15))
    assert(bounds(2) == Sentence("New Sentence", 18, 29))
  }
}
