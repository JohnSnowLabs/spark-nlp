/*
 * Copyright 2017-2022 John Snow Labs
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

package com.johnsnowlabs.nlp.annotators.spell

import com.johnsnowlabs.nlp.annotators.spell.util.Utilities
import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec

class UtilitiesTestSpec extends AnyFlatSpec {

  "levenshteinDistance" should "compute distance between two strings" taggedAs FastTest in {
    val levenshteinDistance = Utilities.levenshteinDistance("hello", "hello")
    assert(levenshteinDistance == 0)
  }

  "Utilities functions" should "work for long words" in {
    val longWord = "hello" * 61
    val reductions = Utilities.reductions(longWord, 3)
    val variants = Utilities.variants(longWord)

    assert(reductions.nonEmpty)
    assert(variants.nonEmpty)
  }

}
