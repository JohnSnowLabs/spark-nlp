/*
 * Copyright 2017-2021 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.annotators.tokenizer.normalizer

import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec

/**
  * tests ported from sacremoses
  * https://github.com/alvations/sacremoses/blob/master/sacremoses/test/test_normalizer.py
  */
class MosesPunctNormalizerTestSpec extends AnyFlatSpec {
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
