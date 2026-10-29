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

  it should "cap at 10 entries without producing a bogus None key" taggedAs FastTest in {
    val tags = (0 until 12).map(i => s"class$i" -> BigInt(i)).toMap
    val scores = Array.tabulate(12)(i => i.toFloat)

    val meta = ViTClassifier.topScoresMetadata(scores, tags)

    assert(meta.length == 10)
    assert(!meta.map(_._1).contains("None"))
  }

  it should "pick the classes that actually scored highest, not an arbitrary 10 by map order" taggedAs FastTest in {
    // Regression test: `tags` is only ever the model's label vocabulary (label -> class index),
    // never sorted by or otherwise related to a specific prediction's scores. Taking `tags.take(10)`
    // (the pre-fix behavior) would silently report whichever 10 classes a Scala Map's hash-based
    // iteration order happened to yield -- the SAME fixed 10 for every input image, unrelated to
    // which classes actually scored highest for THIS image. Found live: benchmarking a real
    // pretrained ViT (1000 ImageNet classes) against 10 known images scored 0% accuracy across the
    // board because the "top" label came from this arbitrary, mostly-irrelevant slice.
    val tags = (0 until 1000).map(i => s"class$i" -> BigInt(i)).toMap
    val scores = Array.tabulate(1000)(i => i.toFloat) // class999 has the highest score, 999.0

    val meta = ViTClassifier.topScoresMetadata(scores, tags)
    val metaMap = meta.toMap

    assert(meta.length == 10)
    val expectedTopLabels = (990 until 1000).map(i => s"class$i").toSet
    assert(meta.map(_._1).toSet == expectedTopLabels)
    assert(metaMap("class999") == "999.0")
  }
}
