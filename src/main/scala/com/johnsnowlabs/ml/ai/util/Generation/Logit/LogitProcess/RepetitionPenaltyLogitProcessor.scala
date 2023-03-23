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

class RepetitionPenaltyLogitProcessor(val penalty: Double) extends LogitProcessor {
  override def call(
      inputIds: Seq[Array[Int]],
      scores: Array[Array[Float]],
      currentLength: Int): Array[Array[Float]] = {
    if (penalty != 1.0) {
      val nextTokenscores = Array.ofDim[Array[Float]](scores.length)

      for (i <- scores.indices) {
        var nextTokenLogit = scores(i)
        val prevInputIds = inputIds.head.distinct
        for ((prevInputId, _) <- prevInputIds.zipWithIndex) {
          var logitPenalty = 1.0
          if (scores(i)(prevInputId) < 0) {
            logitPenalty = this.penalty
          } else {
            logitPenalty = 1 / this.penalty
          }
          nextTokenLogit = nextTokenLogit.updated(
            prevInputId,
            (logitPenalty * nextTokenLogit(prevInputId)).toFloat)
        }
        nextTokenscores(i) = nextTokenLogit
      }
      nextTokenscores
    } else {
      scores
    }
  }
}
