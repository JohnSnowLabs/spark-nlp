/*
 * Copyright 2017 - 2023  John Snow Labs
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package com.johnsnowlabs.ml.ai.util.Generation.Logit.LogitProcess

class MinLengthLogitProcessor(val eosTokenId: Long, val minLength: Int, val vocabSize: Int)
    extends LogitProcessor {
  override def call(
      inputIds: Seq[Array[Long]],
      scores: Array[Array[Float]],
      currentLength: Int): Array[Array[Float]] = {
    if (!eosTokenId.isNaN && currentLength < this.minLength) {
      // create eosTokenId boolean mask
      val isTokenLogit_eosToken =
        for (token <- 0 until this.vocabSize)
          yield if (token == eosTokenId) true else false

      val eosTokenIndices_mask = Array.fill(scores.length)(isTokenLogit_eosToken)

      val newScores =
        for ((nextTokenLogit, bannedTokensIndex_mask) <- scores.zip(eosTokenIndices_mask))
          yield this.setTensorByIndicesToValue(
            nextTokenLogit,
            bannedTokensIndex_mask,
            Float.NegativeInfinity)
      newScores
    } else {
      scores
    }

  }
}
