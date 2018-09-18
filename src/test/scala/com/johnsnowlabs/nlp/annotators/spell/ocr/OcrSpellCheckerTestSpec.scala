package com.johnsnowlabs.nlp.annotators.spell.ocr
import org.scalatest._

class OcrSpellCheckerTestSpec extends FlatSpec {

  trait Scope extends TokenClasses {
    weights += ('l' -> Map('1' -> 0.5f, '!' -> 0.2f), 'P' -> Map('F' -> 0.2f))
  }

  "weighted Levenshtein distance" should "produce weighted results" in new Scope {
    assert(wLevenshteinDist("c1ean", "clean") > wLevenshteinDist("c!ean", "clean"))
    assert(wLevenshteinDist("crean", "clean") > wLevenshteinDist("c!ean", "clean"))
    assert(wLevenshteinDist("Fatient", "Patient") < wLevenshteinDist("Aatient", "Patient"))
  }

  "weighted Levenshtein distance" should "properly compute distance to a regular language - dates" in new Scope {
    assert(wLevenshteinDateDist("10/0772018") == 1.0f)
  }
}
