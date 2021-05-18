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

  def assertTokenization(tokens: Array[String], expected: Array[String]): Assertion =
    assert(tokens.mkString(" ") == expected.mkString(" "))

  "MosesTokenizer" should "encode words correctly" taggedAs FastTest in {
    var text = "This,    is a sentence with weird» symbols… appearing everywhere¿"
    var expected = "This , is a sentence with weird » symbols … appearing everywhere ¿".split(" ")
    assertTokenization(moses.tokenize(text), expected)

    text = "This ain't funny. It's actually hillarious, yet double Ls. | [] < > [ ] & You're gonna shake it off? Don't?"
    expected = Array(
      "This",
      "ain",
      "'t",
      "funny",
      ".",
      "It",
      "'s",
      "actually",
      "hillarious",
      ",",
      "yet",
      "double",
      "Ls",
      ".",
      "|",
      "[",
      "]",
      "<",
      ">",
      "[",
      "]",
      "&",
      "You",
      "'re",
      "gonna",
      "shake",
      "it",
      "off",
      "?",
      "Don",
      "'t",
      "?",
    )
    assertTokenization(moses.tokenize(text), expected)
  }

  "MosesTokenizer" should "tokenize special cases" taggedAs FastTest in {

    var text = "abc def."
    var expected = Array("abc", "def", ".")
    assertTokenization(moses.tokenize(text), expected)

    text = "2016, pp."
    expected = Array("2016", ",", "pp", ".")
    assertTokenization(moses.tokenize(text), expected)

    text = "this 'is' the thing"
    expected = Array("this", "'", "is", "'", "the", "thing")
    assertTokenization(moses.tokenize(text), expected)

    text = "By the mid 1990s a version of the game became a Latvian television series (with a parliamentary setting, and played by Latvian celebrities)."
    expected = "By the mid 1990s a version of the game became a Latvian television series ( with a parliamentary setting , and played by Latvian celebrities ) .".split(" ")
    assertTokenization(moses.tokenize(text), expected)

    text = "'Hello.'"
    expected = "'Hello . '".split(" ")
    assertTokenization(moses.tokenize(text), expected)

    text = "The meeting will take place at 11:00 a.m. Tuesday."
    expected = "The meeting will take place at 11 : 00 a.m. Tuesday .".split(" ")
    assertTokenization(moses.tokenize(text), expected)

  }
  "MosesTokenizer" should "handle multi dots" taggedAs FastTest in {
    val expected = Array("Truncated", "[", "...", "]", "text", "...")
    assertTokenization(moses.tokenize("Truncated [...] text..."), expected)
  }
}
