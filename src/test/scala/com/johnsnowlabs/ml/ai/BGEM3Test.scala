/*
 * Copyright 2017-2024 John Snow Labs
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

import org.scalatest.flatspec.AnyFlatSpec

class BGEM3Test extends AnyFlatSpec {

  import BGEM3._

  "aggregateSparseWeights" should "keep the maximum weight per token id" in {
    // token id 42 appears twice with different weights; the max should win
    val tokens = Array(10, 42, 20, 42)
    val weights = Array(0.1f, 0.3f, 0.2f, 0.9f)
    val result = aggregateSparseWeights(tokens, weights).toMap
    assert(result(42) == 0.9f)
  }

  it should "drop non-positive weights" in {
    val tokens = Array(10, 20, 30)
    val weights = Array(0.5f, 0.0f, -0.1f)
    val result = aggregateSparseWeights(tokens, weights).toMap
    assert(result.contains(10))
    assert(!result.contains(20))
    assert(!result.contains(30))
  }

  it should "exclude special tokens even when their weight is positive" in {
    val tokens =
      Array(SentenceStartTokenId, 10, SentencePadTokenId, SentenceEndTokenId, SentenceUnkTokenId)
    val weights = Array(0.9f, 0.5f, 0.9f, 0.9f, 0.9f)
    val result = aggregateSparseWeights(tokens, weights).toMap
    assert(result == Map(10 -> 0.5f))
  }

  it should "preserve first-occurrence order of token ids" in {
    val tokens = Array(30, 10, 20, 10)
    val weights = Array(0.1f, 0.2f, 0.3f, 0.4f)
    val result = aggregateSparseWeights(tokens, weights)
    assert(result.map(_._1) == Seq(30, 10, 20))
  }

  it should "only consider the overlapping prefix when tokens/weights lengths differ" in {
    // simulates zipping unpadded token ids against a padded (longer) weights row
    val tokens = Array(10, 20)
    val weights = Array(0.5f, 0.6f, 0.7f, 0.8f)
    val result = aggregateSparseWeights(tokens, weights).toMap
    assert(result == Map(10 -> 0.5f, 20 -> 0.6f))
  }

  it should "return an empty sequence when nothing survives filtering" in {
    val tokens = Array(SentenceStartTokenId, SentenceEndTokenId)
    val weights = Array(0.5f, 0.5f)
    assert(aggregateSparseWeights(tokens, weights).isEmpty)
  }

  "expectedSparseWidth" should "return seqLen when the flat length matches batch * seq exactly" in {
    val width = expectedSparseWidth(flatLength = 24, batchSize = 4, seqLen = 6, "token_weights")
    assert(width == 6)
  }

  it should "throw when the flat length doesn't factor into batch * seq" in {
    val ex = intercept[IllegalStateException] {
      expectedSparseWidth(flatLength = 23, batchSize = 4, seqLen = 6, "token_weights")
    }
    assert(ex.getMessage.contains("23"))
    assert(ex.getMessage.contains("24"))
    assert(ex.getMessage.contains("token_weights"))
  }

  it should "throw rather than silently truncate via integer division" in {
    // flatLength=23 / batchSize=4 == 5 (int division), which would have silently produced a
    // wrong width under the old `flatSparse.length / batch.length` logic instead of failing.
    assertThrows[IllegalStateException] {
      expectedSparseWidth(flatLength = 23, batchSize = 4, seqLen = 6, "token_weights")
    }
  }

}
