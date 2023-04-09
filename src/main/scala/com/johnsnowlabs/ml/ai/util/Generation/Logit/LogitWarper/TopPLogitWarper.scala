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
    var scoresUpd = scores
    val scoresShape = Array(scores.length, scores(0).length)
    if (this.p < 1.0) {
      val (sortedscores, sortedIndices) = scores(0).zipWithIndex.sorted.reverse.unzip

      val cumulativeProbs = this.scanLeft(this.softmax(sortedscores))(0.0)(_ + _).drop(1)

      /** Remove tokens with cumulative probability above the threshold (token with 0 are kept) */
      var sortedIndicesToRemove =
        for (prob <- cumulativeProbs)
          yield if (prob > this.p) true else false

      if (minTokensToKeep > 1) {

        /** Keep at least minTokensToKeep (set to minTokensToKeep-1 because we add the first one
          * below)
          */
        sortedIndicesToRemove = List.fill(sortedIndicesToRemove.take(minTokensToKeep).length)(
          false) ++ sortedIndicesToRemove.drop(minTokensToKeep)
      }

      /** Shift the indices to the right to keep also the first token above the threshold */
      sortedIndicesToRemove = sortedIndicesToRemove.takeRight(1) ++ sortedIndicesToRemove
        .dropRight(1)
      sortedIndicesToRemove =
        List.fill(sortedIndicesToRemove.take(1).length)(false) ++ sortedIndicesToRemove
          .drop(1)

      /** scatter sorted tensors to original indexing */
      val indicesToRemove =
        this.scatterValuesOnBatchIndices(sortedIndicesToRemove.toList, sortedIndices)
      scoresUpd =
        for ((nextTokenLogit, indexToRemove) <- scores.zip(
            IndexedSeq.fill(scores.length)(indicesToRemove)))
          yield setTensorByIndicesToValue(
            nextTokenLogit,
            indexToRemove.toIndexedSeq,
            Float.NegativeInfinity)
    }
    scoresUpd
  }

}
