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
import scala.collection.mutable

class NoRepeatNgramsLogitProcessor(val noRepeatNgramSize: Int, val vocabSize: Int)
    extends LogitProcessor {
  override def call(
      inputIds: Seq[Array[Long]],
      scores: Array[Array[Float]],
      currentLength: Int): Array[Array[Float]] = {
    if (noRepeatNgramSize > 0) {

      val bannedTokens =
        calcBannedNgramTokens(inputIds, inputIds.length, this.noRepeatNgramSize, currentLength)
      // create bannedTokens boolean mask
      var bannedTokensIndicesMask = Array.empty[IndexedSeq[Boolean]]
      for (bannedTokensSlice <- bannedTokens) {
        bannedTokensIndicesMask = bannedTokensIndicesMask :+
          (for (token <- 0 until this.vocabSize)
            yield if (bannedTokensSlice.contains(token)) true else false)
      }
      if (!bannedTokensIndicesMask.isEmpty) {
        val nextTokenLogits =
          for ((nextTokenLogit, bannedTokensIndexMask) <- scores.zip(bannedTokensIndicesMask))
            yield setTensorByIndicesToValue(
              nextTokenLogit,
              bannedTokensIndexMask,
              Float.NegativeInfinity)
        nextTokenLogits
      } else {
        scores
      }
    } else {
      scores
    }
  }

  private def calcBannedNgramTokens(
      prevInputIds: Seq[Array[Long]],
      numHypos: Int,
      noRepeatNgramSize: Int,
      curLen: Int): Array[Array[Long]] = {
    // based on fairseq for noRepeatNgram in beam_search
    if (curLen + 1 < noRepeatNgramSize)
      // return no banned tokens if we haven't generated noRepeatNgram_size tokens yet
      return Array.ofDim[Long](numHypos, 0)
    val generatedNgrams =
      Array.tabulate(numHypos)(_ => mutable.Map.empty[IndexedSeq[Long], List[Long]])
    for (idx <- 0 until numHypos) {
      val genTokens = prevInputIds(idx)
      val generatedNgram = generatedNgrams(idx)
      val ngramArrays = for (e <- 0 until noRepeatNgramSize) yield genTokens.drop(e)
      for (ngramInd <- ngramArrays.last.indices) {
        val ngram = for (e <- ngramArrays) yield e(ngramInd)
        val prevNgramTuple = ngram.dropRight(1)
        generatedNgram(prevNgramTuple) =
          generatedNgram.getOrElse(prevNgramTuple, List.empty[Long]) :+ ngram.last
      }
    }
    (for (hypoIdx <- 0 until numHypos)
      yield getGeneratedNgrams(
        prevInputIds,
        generatedNgrams,
        hypoIdx,
        curLen,
        noRepeatNgramSize)).toArray
  }

  private def getGeneratedNgrams(
      prevInputIds: Seq[Array[Long]],
      generatedNgrams: Array[mutable.Map[IndexedSeq[Long], List[Long]]],
      hypoIdx: Int,
      curLen: Int,
      noRepeatNgramSize: Int): Array[Long] = {
    // Before decoding the next token, prevent decoding of ngrams that have already appeared
    val startIdx = curLen + 1 - noRepeatNgramSize
    val ngramIdx = prevInputIds(hypoIdx).slice(startIdx, curLen)
    generatedNgrams(hypoIdx).getOrElse(ngramIdx, List.empty[Long]).toArray
  }
}
