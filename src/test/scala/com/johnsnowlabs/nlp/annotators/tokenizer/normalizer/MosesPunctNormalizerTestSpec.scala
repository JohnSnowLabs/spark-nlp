package com.johnsnowlabs.nlp.annotators.tokenizer.normalizer

import com.johnsnowlabs.tags.FastTest
import org.scalatest.FlatSpec

/**
  * tests ported from sacremoses
  * https://github.com/alvations/sacremoses/blob/master/sacremoses/test/test_normalizer.py
  */
class MosesPunctNormalizerTestSpec extends FlatSpec {
  val normalizer = new MosesPunctNormalizer
  "MosesPunctNormalizer" should "normalize documents" taggedAs FastTest in {
    val documents = Array(
      "The United States in 1805 (color map)                 _Facing_     193",
      "=Formation of the Constitution.=--(1) The plans before the convention,",
      "directions--(1) The infective element must be eliminated. When the ulcer",
      "College of Surgeons, Edinburgh.)]"
    )
    val expected = Array(
      "The United States in 1805 (color map) _Facing_ 193",
      "=Formation of the Constitution.=-- (1) The plans before the convention,",
      "directions-- (1) The infective element must be eliminated. When the ulcer",
      "College of Surgeons, Edinburgh.) ]"
    )
    documents.zip(expected).foreach({ case (doc: String, exp: String) =>
      assert(normalizer.normalize(doc) == exp)
    })
  }

  "MosesPunctNormalizer" should "normalize quote comma" taggedAs FastTest in {
    val text = """THIS EBOOK IS OTHERWISE PROVIDED TO YOU "AS-IS"."""
    val expected = """THIS EBOOK IS OTHERWISE PROVIDED TO YOU "AS-IS.""""
    assert(normalizer.normalize(text) == expected)
  }

  "MosesPunctNormalizer" should "normalize numbers" taggedAs FastTest in {
    var text = "12\u00A0123"
    var expected = "12.123"
    assert(normalizer.normalize(text) == expected)
    text = "12 123"
    expected = text
    assert(normalizer.normalize(text) == expected)

  }

  "MosesPunctNormalizer" should "normalize single apostrophe" taggedAs FastTest in {
    val text = "yesterday ’s reception"
    val expected = "yesterday 's reception"
    assert(normalizer.normalize(text) == expected)
  }

  "MosesPunctNormalizer" should "replace unicode punctation" taggedAs FastTest in {
    var text = "０《１２３》 ４５６％ 【７８９】"
    var expected = """0"123" 456% [789]"""
    assert(normalizer.normalize(text) == expected)

    text = "０《１２３》      ４５６％  '' 【７８９】"
    expected = """0"123" 456% " [789]"""
    assert(normalizer.normalize(text) == expected)

  }
}
