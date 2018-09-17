package com.johnsnowlabs.nlp.annotators.spell.ocr
import org.scalatest._

class OcrSpellCheckerTestSpec extends FlatSpec {

  trait Scope extends TokenClasses {
    weights += ('l' -> Map('1' -> 0.5f, '!' -> 0.2f), 'P' -> Map('F' -> 0.2f))
  }

  "weighted Levenshtein distance" should "produce weighted results" in new Scope {
    assert(wLevenshteinDistance("c1ean", "clean") > wLevenshteinDistance("c!ean", "clean"))
    assert(wLevenshteinDistance("crean", "clean") > wLevenshteinDistance("c!ean", "clean"))
    assert(wLevenshteinDistance("Fatient", "Patient") < wLevenshteinDistance("Aatient", "Patient"))
  }
}
