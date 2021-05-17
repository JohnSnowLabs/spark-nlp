package com.johnsnowlabs.nlp.annotators.tokenizer.moses

import com.johnsnowlabs.nlp.annotators.tokenizer.normalizer.MosesPunctNormalizer
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import org.scalatest.{Assertion, FlatSpec}

/**
  * Tests ported from sacremoses
  */
class MosesTokenizerTestSpec extends FlatSpec {
  val moses = new MosesTokenizer("en")
  val mosesNormalizer = new MosesPunctNormalizer()

  "MosesTokenizer" should "encode words correctly" taggedAs FastTest in {
    val text = "This,    is a sentence with weird» symbols… appearing everywhere¿"
    val expected = "This , is a sentence with weird » symbols … appearing everywhere ¿"

    val tokenized = moses.tokenize(text).mkString(" ")
    assert(tokenized == expected)
  }

  "MosesTokenizer" should "tokenize special cases" taggedAs FastTest in {
    def assertTokenization(tokens: Array[String], expected: Array[String]): Assertion =
      assert(tokens.mkString(" ") == expected.mkString(" "))

    var expected = Array("abc", "def", ".")
    assertTokenization(moses.tokenize("abc def."), expected)

    expected = Array("2016", ",", "pp", ".")
    assertTokenization(moses.tokenize("2016, pp."), expected)

    expected = Array("this", "'", "is", "'", "the", "thing")
    assertTokenization(moses.tokenize("this 'is' the thing"), expected)

    //    expected = Array(" ")
    //    assertTokenization(moses.tokenize("Someone's apostrophe."), expected)
//    println(moses.tokenize("Someone's apostrophe.").mkString("Array(\n  ", ",\n  ", "\n)"))

    //    expected = Array(" ")
    //    assertTokenization(moses.tokenize("Truncated [...] text..."), expected)
//    println(moses.tokenize("Truncated [...] text...").mkString("Array(\n  ", ",\n  ", "\n)"))

  }


}
