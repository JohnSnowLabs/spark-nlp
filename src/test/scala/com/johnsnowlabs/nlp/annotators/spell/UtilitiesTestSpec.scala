package com.johnsnowlabs.nlp.annotators.spell

import com.johnsnowlabs.nlp.annotators.spell.util.Utilities
import org.scalatest.FlatSpec

class UtilitiesTestSpec extends FlatSpec{

  "levenshteinDistance" should "compute distance between two strings" in {
    val levenshteinDistance = Utilities.levenshteinDistance("hello", "hello")
    assert(levenshteinDistance == 0)
  }

}
