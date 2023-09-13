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

import com.johnsnowlabs.ml.ai.util.Generation.Logit.LogitProcess.{
  MinLengthLogitProcessor,
  NoRepeatNgramsLogitProcessor,
  RepetitionPenaltyLogitProcessor
}
import com.johnsnowlabs.ml.ai.util.Generation.Logit.LogitProcessorList
import com.johnsnowlabs.ml.ai.util.Generation.Logit.LogitWarper.{
  TemperatureLogitWarper,
  TopKLogitWarper,
  TopPLogitWarper
}
import com.johnsnowlabs.ml.ai.util.Generation.Search.{BeamScorer, BeamSearchScorer}
import org.tensorflow.{Session, Tensor}

import scala.math._
import scala.util.control.Breaks._

trait Generate {

  /** Text Generation using Beam Search
    *
    * @param inputIds
    *   input ids
    * @param decoderEncoderStateTensors
    *   decoder encoder state tensors
    * @param encoderAttentionMaskTensors
    *   encoder attention mask tensors
    * @param decoderInputs
    *   decoder inputs
    * @param maxOutputLength
    *   max output length
    * @param minOutputLength
    *   min output length
    * @param doSample
    *   do sample
    * @param beamSize
    *   beam size
    * @param numReturnSequences
    *   num return sequences
    * @param temperature
    *   temperature
    * @param topK
    *   top K
    * @param topP
    *   top P
    * @param repetitionPenalty
    *   repetition penalty
    * @param noRepeatNgramSize
    *   no repeat ngram size
    * @param vocabSize
    *   vocab size
    * @param eosTokenId
    *   eos token id
    * @param paddingTokenId
    *   padding token id
    * @param randomSeed
    *   random seed
    * @param ignoreTokenIds
    *   ignore token ids
    * @param session
    *   session
    * @return
    *   Array of generated sequences
    */
  def generate(
      inputIds: Seq[Array[Int]],
      decoderEncoderStateTensors: Tensor,
      encoderAttentionMaskTensors: Tensor,
      decoderInputs: Array[Array[Int]],
      maxOutputLength: Int,
      minOutputLength: Int,
      doSample: Boolean,
      beamSize: Int,
      numReturnSequences: Int,
      temperature: Double,
      topK: Int,
      topP: Double,
      repetitionPenalty: Double,
      noRepeatNgramSize: Int,
      vocabSize: Int,
      eosTokenId: Int,
      paddingTokenId: Int,
      randomSeed: Option[Long],
      ignoreTokenIds: Array[Int] = Array(),
      session: Session,
      applySoftmax: Boolean = true): Array[Array[Int]] = {

    // TODO: Add support for ignoreTokenIds

    val logitProcessorList = new LogitProcessorList()

    logitProcessorList.addProcess(new RepetitionPenaltyLogitProcessor(repetitionPenalty))

    logitProcessorList.addProcess(
      new NoRepeatNgramsLogitProcessor(
        noRepeatNgramSize = noRepeatNgramSize,
        vocabSize = vocabSize))

    logitProcessorList.addProcess(
      new MinLengthLogitProcessor(eosTokenId, minOutputLength, vocabSize))

    logitProcessorList.addProcess(new TemperatureLogitWarper(temperature))

    logitProcessorList.addProcess(new TopKLogitWarper(topK))

    logitProcessorList.addProcess(new TopPLogitWarper(topP))

    val beamSearchScorer = new BeamSearchScorer(
      beamSize = beamSize,
      batchSize = inputIds.length,
      lengthPenalty = repetitionPenalty.toFloat,
      doEarlyStopping = false,
      numBeamHypothesisToKeep = numReturnSequences,
      maxLength = maxOutputLength)

    this.beamSearch(
      inputIds,
      decoderInputs,
      decoderEncoderStateTensors,
      encoderAttentionMaskTensors,
      beamSearchScorer,
      logitProcessorList,
      maxOutputLength,
      paddingTokenId,
      eosTokenId,
      doSample,
      randomSeed,
      session,
      applySoftmax)
  }

  /** Beam Search for text generation
    *
    * @param encoderInputIdsVals
    *   encoder input ids vals
    * @param inputIdsVal
    *   input ids val
    * @param decoderEncoderStateTensors
    *   decoder encoder state tensors
    * @param encoderAttentionMaskTensors
    *   encoder attention mask tensors
    * @param beamScorer
    *   beam scorer
    * @param logitProcessor
    *   logit processor
    * @param maxLength
    *   max length
    * @param padTokenId
    *   pad token id
    * @param eosTokenId
    *   eos token id
    * @param doSample
    *   do sample
    * @param randomSeed
    *   random seed
    * @param session
    *   session
    * @return
    */
  def beamSearch(
      encoderInputIdsVals: Seq[Array[Int]],
      inputIdsVal: Seq[Array[Int]],
      decoderEncoderStateTensors: Tensor,
      encoderAttentionMaskTensors: Tensor,
      beamScorer: BeamScorer,
      logitProcessor: LogitProcessorList,
      maxLength: Int,
      padTokenId: Int,
      eosTokenId: Int,
      doSample: Boolean,
      randomSeed: Option[Long],
      session: Session,
      applySoftmax: Boolean): Array[Array[Int]] = {
    val inputIds = inputIdsVal
    val batchSize = beamScorer.getBeamHypothesesSeq.length
    val numBeams = beamScorer.getNumBeams
    val batchBeamSize = batchSize * numBeams
    var currentLength = inputIds.head.length

    var beamScores = Array.ofDim[Float](batchSize * numBeams)
    beamScores = beamScores.zipWithIndex.map { case (_, ind) =>
      if (ind % numBeams == 0) 0 else (-1e+9).toFloat
    }
    var beamIndices = Seq.fill(batchBeamSize)(Array[Int]())
    var nextIndices = Array[Array[Int]]()
    var nextTokens = Array[Array[Int]]()
    var expandedInputs = inputIds.flatMap(x => List.fill(numBeams)(x))
    val expandedEncoderInputIdsVals = encoderInputIdsVals.flatMap(x => List.fill(numBeams)(x))
    breakable {
      while (true) {

        // Feed the encoder input ids and decoder input ids to the model and get the output
        // return shape (beamSize,vocabSize)
        val nextTokenLogits =
          this.getModelOutput(
            expandedEncoderInputIdsVals,
            expandedInputs,
            decoderEncoderStateTensors,
            encoderAttentionMaskTensors,
            maxLength,
            session)

        // Optionally Apply log softmax to model outputs
        var nextTokenScores =
          if (applySoftmax) nextTokenLogits.map(logSoftmax) else nextTokenLogits

        // Process the logits by defined logit processors
        val nextTokenScoresProcessed =
          logitProcessor.process(expandedInputs, nextTokenScores, currentLength)

        // Add previous beam scores to the output
        nextTokenScores = nextTokenScoresProcessed.zipWithIndex.map { case (x, ind1) =>
          x.zipWithIndex.map { case (y, _) =>
            y + beamScores(ind1)
          }
        }
        // Process the logits by defined logit warpers
        if (doSample) {
          nextTokenScores = logitProcessor.warp(expandedInputs, nextTokenScores, currentLength)
        }
        // Reshape next token score to (batchSize, vocabSize * numBeams)
        val vocabSize = nextTokenScores.head.length
        val reshapedNextTokenScores =
          reshapeArray(nextTokenScores, batchSize, vocabSize * numBeams)

        nextTokenScores = reshapedNextTokenScores

        var nextKTopTokenScores: Array[Array[Float]] = Array[Array[Float]]()
        var nextKTopTokens: Array[Array[Int]] = Array[Array[Int]]()

        if (doSample) {
          val nextKIndices = nextTokenScores.map(x => {
            multinomialSampling(x, 2 * numBeams, randomSeed)
          })
          nextKTopTokenScores = Array.ofDim[Float](nextKIndices.length, nextKIndices.head.length)
          for (i <- nextKIndices.indices) {
            for (j <- nextKIndices(i).indices) {
              nextKTopTokenScores(i)(j) = nextTokenScores(i)(nextKIndices(i)(j))
            }
          }
          nextKTopTokenScores =
            nextKTopTokenScores.map(x => x.zipWithIndex.sortWith(_._1 > _._1).map(_._1))
          val tempNextKInd =
            nextKTopTokenScores.map(x => x.zipWithIndex.sortWith(_._1 > _._1).map(_._2))
          nextKTopTokens = Array.ofDim[Int](nextKIndices.length, nextKIndices.head.length)

          for (i <- tempNextKInd.indices) {
            for (j <- tempNextKInd(i).indices) {
              nextKTopTokens(i)(j) = nextKIndices(i)(tempNextKInd(i)(j))
            }
          }
        } else {
          nextKTopTokenScores = nextTokenScores.map(x =>
            x.zipWithIndex.sortWith(_._1 > _._1).take(2 * numBeams).map(_._1))
          nextKTopTokens = nextTokenScores.map(x =>
            x.zipWithIndex.sortWith(_._1 > _._1).take(2 * numBeams).map(_._2))
        }
        nextIndices = nextKTopTokens.map(y => y.map(x => x / vocabSize))
        nextTokens = nextKTopTokens.map(y => y.map(x => x % vocabSize))

        val beamOutputs = beamScorer.process(
          expandedInputs,
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
        var newInputIds = Seq[Array[Int]]()

        for ((i, ind) <- beamIdx.zipWithIndex) {
          val tempInput = expandedInputs(i) :+ beamNextTokens(ind)
          newInputIds = newInputIds :+ tempInput
        }
        expandedInputs = newInputIds
        beamScores = newBeamScores
        beamIndices = beamIndices.indices.map { elem =>
          beamIndices(beamIdx(elem)) :+ beamIdx(elem)
        }
        currentLength = currentLength + 1
        if (beamScorer.isDone || (expandedInputs.head.length >= maxLength)) {
          break

        }
      }
    }

    val sequenceOutputs = beamScorer.finalize(
      inputIds = expandedInputs,
      finalBeamScores = beamScores,
      finalBeamTokens = nextTokens.flatMap(_.toList),
      finalBeamIndices = nextIndices.flatMap(_.toList),
      maxLength = maxLength,
      padTokenId = padTokenId,
      eosTokenId = eosTokenId,
      beamIndices = beamIndices)
    sequenceOutputs._1
  }

  def logSoftmax(values: Array[Float]): Array[Float] = {
    val c = values.max
    val expElem = values.map(x => exp(x - c))
    val logSumExp = log(expElem.sum)
    values.map(x => (x - c - logSumExp).toFloat)
  }

  /** Reshapes a 1D array into a 2D array with the specified number of rows and columns.
    *
    * @param inputArray
    *   The input array to reshape
    * @param numRows
    *   The number of rows in the output array
    * @param numCols
    *   The number of columns in the output array
    * @return
    *   The reshaped array
    */
  def reshapeArray(
      inputArray: Array[Array[Float]],
      numRows: Int,
      numCols: Int): Array[Array[Float]] = {
    if (inputArray.length * inputArray(0).length != numRows * numCols) {
      throw new IllegalArgumentException(
        "Number of elements in input array does not match desired shape")
    }

    val flatArray = inputArray.flatten // Flatten the input array into a 1D array
    val outputArray = Array.ofDim[Float](numRows, numCols) // Initialize the output array

    // Loop through the output array and fill it with elements from the flat array
    for (i <- 0 until numRows) {
      for (j <- 0 until numCols) {
        outputArray(i)(j) = flatArray(i * numCols + j)
      }
    }

    outputArray // Return the reshaped array
  }

  /** Samples from a multinomial distribution using the provided logits.
    *
    * @param logitValues
    *   The logits to sample from
    * @param k
    *   The number of samples to draw
    * @param seed
    *   The random seed to use
    * @return
    *   The sampled indices
    */
  def multinomialSampling(logitValues: Array[Float], k: Int, seed: Option[Long]): Array[Int] = {
    val (distFiltered, indices) =
      logitValues.zipWithIndex.filter { case (elem, index) => !elem.isInfinite }.sorted.unzip
    if (!distFiltered.isEmpty) {

      val maxLogit = distFiltered.max
      val expLogitValues = distFiltered.map(logit => math.exp(logit - maxLogit))
      val sumExpLogitValues = expLogitValues.sum
      val probabilities = expLogitValues.map(_ / sumExpLogitValues)

      val selectedIndices = new Array[Int](k)
      var seededRandom = new scala.util.Random()
      if (seed.isDefined) {
        seededRandom = new scala.util.Random(seed.get)
      }
      for (i <- 0 until k) {
        var rand = scala.util.Random.nextDouble()
        if (seed.isDefined) {
          rand = new scala.util.Random(seed.get).nextDouble()
        }
        var cumProb = 0.0
        var j = 0
        while (j < probabilities.length - i) {
          cumProb += probabilities(j)
          if (rand < cumProb) {
            selectedIndices(i) = indices(j)
            probabilities(j) = 0.0
            indices(j) = indices(indices.length - i - 1)
            j = probabilities.length
          }
          j += 1
        }
      }

      selectedIndices
    } else {
      val selectedIndices = new Array[Int](k)
      for (i <- 0 until k) {
        selectedIndices(i) = 0
      }
      selectedIndices
    }
  }

  /** Calls the model and returns the output logits.
    *
    * @param encoderInputIds
    *   Input IDs for the Encoder
    * @param decoderInputIds
    *   Input IDs for the Decoder
    * @param decoderEncoderStateTensors
    *   Tensor of encoded input for the decoder
    * @param encoderAttentionMaskTensors
    *   Tensor for encoder attention mask
    * @param maxLength
    *   Max length of the input
    * @param session
    *   Tensorflow Session
    * @return
    *   Logits for the input
    */
  def getModelOutput(
      encoderInputIds: Seq[Array[Int]],
      decoderInputIds: Seq[Array[Int]],
      decoderEncoderStateTensors: Tensor,
      encoderAttentionMaskTensors: Tensor,
      maxLength: Int,
      session: Session): Array[Array[Float]]

  /** Samples from a multinomial distribution using the provided logits.
    *
    * @param logits
    *   The logits to sample from
    * @param k
    *   The number of samples to draw
    * @param seed
    *   The random seed to use
    * @return
    *   The sampled indices
    */
  def sample(logits: Seq[Float], k: Int, seed: Long = 42): Array[Int] = {
    val maxLogit = logits.max
    val logitsExp = logits.map(logit => math.exp(logit - maxLogit))
    val sumExp = logitsExp.sum
    val probs = logitsExp.map(exp => exp / sumExp)
    val SeededRandom = new scala.util.Random(seed)
    val randSeq = Seq.fill(k)(SeededRandom.nextDouble())
    var cumProb = 0.0
    var index = 0
    var results = Seq[Int]()
    for (rand <- randSeq) {
      while (index < probs.length - 1 && cumProb + probs(index) < rand) {
        cumProb += probs(index)
        index += 1
      }
      results :+= index
    }
    results.toArray
  }

  def softmax(logitValues: Array[Float]): Array[Float] = {
    val maxLogit = logitValues.max
    val logitsExp = logitValues.map(l => Math.exp(l - maxLogit))
    val expSum = logitsExp.sum
    logitsExp.map(exp => (exp / expSum).toFloat)
  }

  def getCDF(probs: Array[Float]): Array[Float] = {
    val cdf = Array.ofDim[Float](probs.length)
    var sum = 0.0
    for (i <- probs.indices) {
      sum += probs(i)
      cdf(i) = sum.toFloat
    }
    cdf
  }

}
