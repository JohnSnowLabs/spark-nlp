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

package com.johnsnowlabs.ml.ai.util.Generation.Logit.LogitWarper

class TopKLogitWarper(
    val k: Int,
    val filterValue: Float = Float.NegativeInfinity,
    val minTokensToKeep: Int = 1)
    extends LogitWarper {
  override def call(
      inputIds: Seq[Array[Int]],
      scores: Array[Array[Float]],
      currentLength: Int): Array[Array[Float]] = {
    var logitsUpd = scores
    val logitsShape = Array(scores.length, scores(0).length)
    if (k > 0) {
      val topKup = k.max(minTokensToKeep).min(logitsShape.last) // Safety check

      /** Remove all tokens with a probability less than the last token of the top-k */
      val removeLimit = scores(0).sortWith(_ > _).take(topKup).min
      val indicesToRemove =
        for (logit <- scores)
          yield for (elem <- logit) yield if (elem < removeLimit) true else false

      logitsUpd =
        for ((nextTokenLogit, indexToRemove) <- scores.zip(indicesToRemove))
          yield this.setTensorByIndicesToValue(
            nextTokenLogit,
            indexToRemove,
            Float.NegativeInfinity)
    }
    logitsUpd
  }
}
