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
    val logitsUpd = scores.map(_.clone()) // Deep copy of the scores

    if (p < 1.0) {
      val scoresFiltered = scores // Filter out infinite values
      val scoresSoftmaxed = scoresFiltered.map(softmax) // Softmax the scores

      for ((logits, i) <- scoresSoftmaxed.zipWithIndex) {
        val topPIndices = getTopPIndices(logits, p)
        // Mask the values that are not in the top-p
        val maskedValues = maskNotTopPValues(logitsUpd(i), topPIndices)
        logitsUpd(i) = maskedValues
      }
    }

    logitsUpd
  }

  private def getTopPIndices(logits: Array[Float], p: Double): Array[Int] = {
    // sort the logits in descending order
    var sortedLogits = logits.zipWithIndex.sortBy(-_._1)

    // filter out the negative infinity values
    sortedLogits = sortedLogits.filter(_._1 > 0.0)

    // cumulative sum of the probabilities
    val cumSum = sortedLogits.map(_._1).scanLeft(0.0)(_ + _)

    // find the index of the last element that is less than p
    val lastIdx = cumSum.indexWhere(_ >= p)
    // if the last index is less than the minimum tokens to keep, return the top p tokens

    if (lastIdx < minTokensToKeep) {
      sortedLogits.take(math.ceil(p * logits.length).toInt).map(_._2)
    } else {
      sortedLogits.take(lastIdx).map(_._2)
    }

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
