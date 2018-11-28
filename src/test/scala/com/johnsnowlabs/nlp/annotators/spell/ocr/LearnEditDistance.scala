package com.johnsnowlabs.nlp.annotators.spell.ocr

import org.scalatest.FlatSpec

class LearnEditDistance extends FlatSpec {

  /* TODO  Note some test cases should be moved to internal repo */

  trait Scope extends WeightedLevenshtein {
    val weights = Map('l' -> Map('1' -> 0.5f, '!' -> 0.2f), 'P' -> Map('F' -> 0.2f))
  }

  "learn edit distance" should "learn edit operations from corpus" in new Scope {
    // learn deletion
    var edits = learnDist("actress", "atress")

    // learn deletion followed by replacement
    edits = learnDist("actress", "atres1")

    // learn insertion
    edits = learnDist("actress", "actreoss")

    // learn replacement + insertion
    edits = learnDist("actress", "aotreoss")

  }

}
