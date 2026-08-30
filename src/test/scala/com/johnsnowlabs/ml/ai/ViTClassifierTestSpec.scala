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

package com.johnsnowlabs.ml.ai

import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec

class ViTClassifierTestSpec extends AnyFlatSpec {

  "ViTClassifier.topScoresMetadata" should "key each entry by the real label, not Option#toString" taggedAs FastTest in {
    val tags = Map("cat" -> BigInt(0), "dog" -> BigInt(1), "bird" -> BigInt(2))
    val scores = Array(0.1f, 0.7f, 0.2f)

    val meta = ViTClassifier.topScoresMetadata(scores, tags)

    assert(meta.toMap == Map("cat" -> "0.1", "dog" -> "0.7", "bird" -> "0.2"))
    assert(meta.map(_._1).forall(key => key != "None" && !key.startsWith("Some(")))
  }

  it should "drop scores whose index isn't among the first 10 tags, instead of a bogus None key" taggedAs FastTest in {
    val tags = (0 until 12).map(i => s"class$i" -> BigInt(i)).toMap
    val scores = Array.tabulate(12)(i => i.toFloat)

    val meta = ViTClassifier.topScoresMetadata(scores, tags)

    assert(meta.length == 10)
    assert(!meta.map(_._1).contains("None"))
  }
}
