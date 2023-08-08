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

package com.johnsnowlabs.ml.ai.util.Generation.Search

import scala.util.control.Breaks._

class BeamSearchScorer(
    var beamSize: Int,
    var batchSize: Int,
    var lengthPenalty: Float = 1.0f,
    var doEarlyStopping: Boolean = false,
    var numBeamHypothesisToKeep: Int = 1,
    var maxLength: Int)
    extends BeamScorer {

  val numBeams: Int = beamSize
  private var beamHypothesesSeq: Seq[BeamHypotheses] = Seq.empty[BeamHypotheses]
  (1 to batchSize) foreach (i =>
    beamHypothesesSeq = beamHypothesesSeq :+ new BeamHypotheses(
      lengthPenalty = lengthPenalty,
      numBeams = beamSize,
      earlyStopping = doEarlyStopping,
      maxLength = maxLength))

  override def getBeamHypothesesSeq: Seq[BeamHypotheses] = {
    beamHypothesesSeq
  }

  override def getNumBeams: Int = numBeams
  private val done: Array[Boolean] = Array.fill(batchSize)(false)

  override def process(
      inputIds: Seq[Array[Int]],
      nextScores: Seq[Array[Float]],
      nextTokens: Seq[Array[Int]],
      nextIndices: Seq[Array[Int]],
      padTokenId: Int,
      eosTokenId: Int,
      beamIndices: Seq[Array[Int]],
      currentLength: Int): (Array[Array[Float]], Array[Array[Int]], Array[Array[Int]]) = {
//    val currentLength = inputIds.length
    val batchSize = this.beamHypothesesSeq.length
    val nextBeamScores = Array.ofDim[Float](batchSize, this.beamSize)
    val nextBeamTokens = Array.ofDim[Int](batchSize, this.beamSize)
    val nextBeamIndices = Array.ofDim[Int](batchSize, this.beamSize)

    this.beamHypothesesSeq.zipWithIndex.foreach { case (hypotheses, batchIdx) =>
      breakable {
        if (this.done(batchIdx)) {
          nextBeamScores(batchIdx) = nextBeamScores(batchIdx).map(_ => 0.0f)
          nextBeamTokens(batchIdx) = nextBeamTokens(batchIdx).map(_ => padTokenId)
          nextBeamIndices(batchIdx) = nextBeamIndices(batchIdx).map(_ => 0)
          break
        }
        var beamIdx = 0
        var beamTokenRank = 0
        while (beamTokenRank < nextScores(batchIdx).length && beamIdx < this.beamSize) {

          val nextScore = nextScores(batchIdx)(beamTokenRank)
          val nextToken = nextTokens(batchIdx)(beamTokenRank)
          val nextIndex = nextIndices(batchIdx)(beamTokenRank)
          val batchBeamIdx = batchIdx * this.beamSize + nextIndex

          if (eosTokenId == nextToken) {
            if (beamTokenRank >= this.beamSize) {
              break
            }
            var beamIndex = Array[Int]()
            if (beamIndices.nonEmpty) {
              beamIndex = beamIndices(batchBeamIdx)
              beamIndex = beamIndex.map(i => i + batchBeamIdx)
            }

            hypotheses.add(inputIds(batchBeamIdx), nextScore, beamIndex)
          } else {
            nextBeamScores(batchIdx)(beamIdx) = nextScore
            nextBeamTokens(batchIdx)(beamIdx) = nextToken
            nextBeamIndices(batchIdx)(beamIdx) = batchBeamIdx
            beamIdx += 1
          }
          beamTokenRank += 1
        }
        this.done(batchIdx) =
          this.done(batchIdx) || hypotheses.isDone(nextScores(batchIdx).max, currentLength)
      }
    }
    (nextBeamScores, nextBeamTokens, nextBeamIndices)
  }

  override def finalize(
      inputIds: Seq[Array[Int]],
      finalBeamScores: Array[Float],
      finalBeamTokens: Array[Int],
      finalBeamIndices: Array[Int],
      maxLength: Int,
      padTokenId: Int,
      eosTokenId: Int,
      beamIndices: Seq[Array[Int]]): (Array[Array[Int]], Array[Float], Array[Array[Int]]) = {
    val batchSize = this.beamHypothesesSeq.length
    this.beamHypothesesSeq.zipWithIndex.foreach { case (hypotheses, batchIdx) =>
      breakable {
        if (!this.done.contains(false)) {
          break
        }
        for (beamId <- 0 until this.beamSize) {
          var batchBeamIdx = batchIdx * this.beamSize + beamId
          var finalScore = finalBeamScores(batchBeamIdx)
          var finalTokens = inputIds(batchBeamIdx)
          var beamIndex = Array[Int]()
          if (beamIndices.nonEmpty) {
            beamIndex = beamIndices(batchBeamIdx)
          }
          hypotheses.add(finalTokens, finalScore, beamIndex)
        }
      }
    }
    val sentLengths = Array.ofDim[Int](batchSize * this.numBeamHypothesisToKeep)
    var best = Seq[Array[Int]]()
    var bestIndices = Seq[Array[Int]]()
    val bestScores = Array.ofDim[Float](batchSize * this.numBeamHypothesisToKeep)
    this.beamHypothesesSeq.zipWithIndex.foreach { case (hypotheses, i) =>
      breakable {
        var sortedHypotheses = hypotheses.getBeams.sortWith(_._1 < _._1)
        for (j <- 0 until this.numBeamHypothesisToKeep) {
          val bestHypothesisTuple = sortedHypotheses.last
          val bestScore = bestHypothesisTuple._1
          val bestHypothesis = bestHypothesisTuple._2
          val bestIndex = bestHypothesisTuple._3
          sentLengths(this.numBeamHypothesisToKeep * i + j) = bestHypothesis.length
          best = best :+ bestHypothesis
          bestIndices = bestIndices :+ bestIndex
          bestScores(i * this.numBeamHypothesisToKeep + j) = bestScore.toFloat
          sortedHypotheses = sortedHypotheses.dropRight(1)
        }
      }
    }
    val sentLengthMax = sentLengths.max + 1
    val sentMaxLength = Math.min(sentLengthMax, maxLength) + 1
    var decoded =
      Array.fill[Int](batchSize * this.numBeamHypothesisToKeep, sentMaxLength)(padTokenId)
    var indices = Array.ofDim[Int](batchSize * this.numBeamHypothesisToKeep, sentMaxLength)

    if (sentLengths.min != sentLengths.max) {
      decoded = decoded.map(each => each.map(_ => padTokenId))
    }
    indices = indices.map(each => each.map(_ => -1))
    for (i <- best.indices) {
      val hypo = best(i)
      val bestIdx = bestIndices(i)
      for (j <- 0 until sentLengths(i)) {
        decoded(i)(j) = hypo(j)
//        indices(i)(j) = bestIdx(j)
      }
      if (sentLengths(i) < sentMaxLength) {
        decoded(i)(sentLengths(i)) = eosTokenId
      }
    }
    (decoded, bestScores, indices)

  }

  override def isDone: Boolean = !this.done.contains(false)
}
