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

package com.johnsnowlabs.ml.ai.util.Generation
import com.johnsnowlabs.ml.ai.util.Generation.Search.BeamScorer
import com.johnsnowlabs.ml.ai.util.Generation.Logit.LogitProcessorList
import scala.math.*
import scala.util.control.Breaks.*

trait Generate {
  def beamSearch(
      inputIdsVal: Seq[Array[Int]],
      beamScorer: BeamScorer,
      logitProcessor: LogitProcessorList,
      maxLength: Int,
      padTokenId: Int,
      eosTokenId: Int): Array[Array[Int]] = {
    var inputIds = inputIdsVal
    val batchSize = beamScorer.getBeamHypothesesSeq.length
    val numBeams = beamScorer.getNumBeams
    val batchBeamSize = inputIds.length
    var currentLength = inputIds.head.length

    if (numBeams * batchSize != batchBeamSize) {
      throw new Exception(
        "Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}.")
    }

    //    var beamScores = Array.ofDim[Double](batchSize, numBeams)
    //    beamScores = beamScores.map(x =>
    //      x.zipWithIndex.map { case (_, ind) => { if (ind % numBeams == 0) 0 else -1e-9 } })
    var beamScores = Array.ofDim[Float](batchSize * numBeams)
    beamScores = beamScores.zipWithIndex.map { case (_, ind) =>
      if (ind % numBeams == 0) 0 else (-1e-9).toFloat
    }
    var beamIndices = Seq.fill(batchBeamSize)(Array[Int]())
    var nextIndices = Array[Array[Int]]()
    var nextTokens = Array[Array[Int]]()
    while (true) {
      breakable {
        var expandedInputs = inputIds.flatMap(x => List.fill(numBeams)(x))
        var nextTokenLogits = this.getModelOutput(expandedInputs)
        var nextTokenScores = nextTokenLogits.map(logSoftmax)
        var nextTokenScoresProcessed =
          logitProcessor.process(inputIds, nextTokenScores, currentLength)
        nextTokenScores = nextTokenScoresProcessed.zipWithIndex.map { case (x, ind1) =>
          x.zipWithIndex.map { case (y, _) =>
            (y + beamScores(ind1)).toFloat
          }
        }
        val vocabSize = nextTokenScores.head.length
        var reshapedNextTokenScores = Array.ofDim[Float](batchSize, vocabSize * numBeams)
        for (i <- 0 until batchSize * numBeams by numBeams) {
          var tempScores = Seq[Float]()
          for (j <- i until i + numBeams) {
            tempScores ++= nextTokenScores(j).toSeq
          }
          reshapedNextTokenScores((i / numBeams)) = tempScores.toArray
        }
        nextTokenScores = reshapedNextTokenScores
        val nextKTopTokenScores: Array[Array[Float]] =
          nextTokenScores.map(x => x.zipWithIndex.sortBy(-_._1).take(2 * numBeams).map(_._1))
        val nextKTopTokens: Array[Array[Int]] =
          nextTokenScores.map(x => x.zipWithIndex.sortBy(-_._1).take(2 * numBeams).map(_._2))

        nextIndices = nextKTopTokens.map(y => y.map(x => x / vocabSize))
        nextTokens = nextKTopTokens.map(y => y.map(x => x % vocabSize))

        var beamOutputs = beamScorer.process(
          inputIds,
          nextKTopTokenScores,
          nextTokens,
          nextIndices,
          padTokenId,
          eosTokenId,
          beamIndices,
          currentLength)
        val newBeamScores = beamOutputs._1.flatMap(_.toList)
        val beamNextTokens = beamOutputs._2.flatMap(_.toList)
        val beamIdx = beamOutputs._3.flatMap(_.toList)
        val newInputIds = Seq()

        for ((i, ind) <- beamIdx.zipWithIndex) {
          val tempInput = inputIds(i)
          tempInput :+ beamNextTokens(ind)
          newInputIds :+ tempInput
        }
        inputIds = newInputIds
        beamScores = newBeamScores
        currentLength = currentLength + 1
        if (beamScorer.isDone) {
          break

        }
      }
    }

    var sequenceOutputs = beamScorer.finalize(
      inputIds=inputIds,
      finalBeamScores=beamScores,
      finalBeamTokens=nextTokens.flatMap(_.toList),
      finalBeamIndices=nextIndices.flatMap(_.toList),
      maxLength=maxLength,
      padTokenId=padTokenId,
      eosTokenId=eosTokenId,
      beamIndices=beamIndices)
    sequenceOutputs._1
  }

  def getModelOutput(inputIds: Seq[Array[Int]]): Array[Array[Float]]

  def logSoftmax(values: Array[Float]): Array[Float] = {
    val c = values.max
    val expElem = values.map(x => exp(x - c))
    val logSumExp = log(expElem.sum)
    values.map(x => (x - c - logSumExp).toFloat)
  }
}
