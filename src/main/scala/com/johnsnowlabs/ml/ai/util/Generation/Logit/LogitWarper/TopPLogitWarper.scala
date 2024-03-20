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

class TopPLogitWarper(val p: Double, val minTokensToKeep: Int = 1) extends LogitWarper {
  override def call(
      inputIds: Seq[Array[Int]],
      scores: Array[Array[Float]],
      currentLength: Int): Array[Array[Float]] = {
    var logitsUpd = scores.clone() // Make a copy to update

    if (p < 1.0) {
      val scoresFiltered = scores.map(_.filterNot(_.isInfinite)) // Filter out infinite values
      val scoresShape = Array(scoresFiltered.length, scoresFiltered(0).length)
      val topPup = math.ceil(p * scoresShape.last).toInt // Determine top-p threshold

      for ((logits, i) <- scores.zipWithIndex) {
        val topPIndices = getTopPIndices(logits, topPup)
        val maskedValues = maskNotTopPValues(logits, topPIndices)
        logitsUpd(i) = maskedValues
      }
    }

    logitsUpd
  }

  private def getTopPIndices(logits: Array[Float], k: Int): Array[Int] = {
    logits.indices.sortBy(logits(_)).reverse.take(k).toArray
  }

  private def maskNotTopPValues(logits: Array[Float], topPIndices: Array[Int]): Array[Float] = {
    val maskedValues = logits.clone()
    for (i <- logits.indices) {
      if (!topPIndices.contains(i)) {
        maskedValues(i) = Float.NegativeInfinity
      }
    }
    maskedValues
  }
}
