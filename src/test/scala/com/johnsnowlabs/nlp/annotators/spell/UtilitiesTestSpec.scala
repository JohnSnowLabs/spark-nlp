package com.johnsnowlabs.nlp.annotators.spell

import com.johnsnowlabs.nlp.annotators.spell.util.Utilities
import com.johnsnowlabs.tags.FastTest
import org.scalatest.FlatSpec

class UtilitiesTestSpec extends FlatSpec{

  "levenshteinDistance" should "compute distance between two strings" taggedAs FastTest in {
    val levenshteinDistance = Utilities.levenshteinDistance("hello", "hello")
    assert(levenshteinDistance == 0)
  }

}
