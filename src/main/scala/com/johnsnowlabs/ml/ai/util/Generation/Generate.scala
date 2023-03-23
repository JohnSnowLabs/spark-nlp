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
import scala.math._
import scala.util.control.Breaks._
import scala.util.Random

trait Generate {
  def beamSearch(
      encoderInputIdsVals: Seq[Array[Int]],
      inputIdsVal: Seq[Array[Int]],
      beamScorer: BeamScorer,
      logitProcessor: LogitProcessorList,
      maxLength: Int,
      padTokenId: Int,
      eosTokenId: Int,
      doSample: Boolean,
      randomSeed: Long): Array[Array[Int]] = {
    var inputIds = inputIdsVal
    val batchSize = beamScorer.getBeamHypothesesSeq.length
    val numBeams = beamScorer.getNumBeams
    val batchBeamSize = batchSize * numBeams
    var currentLength = inputIds.head.length

//    if (numBeams * batchSize != batchBeamSize) {
//      throw new Exception(
//        "Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}.")
//    }

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
    var expandedInputs = inputIds.flatMap(x => List.fill(numBeams)(x))
    val expandedEncoderInputIdsVals = encoderInputIdsVals.flatMap(x => List.fill(numBeams)(x))
    breakable {
      while (true) {

        // Feed the encoder input ids and decoder input ids to the model and get the output
        // return shape (beamSize,vocabSize)
        val nextTokenLogits =
          this.getModelOutput(expandedEncoderInputIdsVals, expandedInputs, maxLength)

        // Apply log softmax to model outputs
        var nextTokenScores = nextTokenLogits.map(logSoftmax)

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
        nextTokenScores = logitProcessor.warp(expandedInputs, nextTokenScores, currentLength)

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
          newInputIds = newInputIds :+ (tempInput)
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

  def getModelOutput(
      encoderInputIds: Seq[Array[Int]],
      decoderInputIds: Seq[Array[Int]],
      maxLength: Int): Array[Array[Float]]

  def logSoftmax(values: Array[Float]): Array[Float] = {
    val c = values.max
    val expElem = values.map(x => exp(x - c))
    val logSumExp = log(expElem.sum)
    values.map(x => (x - c - logSumExp).toFloat)
  }

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

  def multinomialSampling(logitValues: Array[Float], k: Int, seed: Long = 42): Array[Int] = {
    val (distFiltered, indices) =
      logitValues.zipWithIndex.filter { case (elem, index) => !elem.isInfinite }.sorted.unzip

    val maxLogit = distFiltered.max
    val expLogitValues = distFiltered.map(logit => math.exp(logit - maxLogit))
    val sumExpLogitValues = expLogitValues.sum
    val probabilities = expLogitValues.map(_ / sumExpLogitValues)

//    val indices = Array.range(0, logitValues.length)
    val selectedIndices = new Array[Int](k)
    val seededRandom = new scala.util.Random(seed)
    for (i <- 0 until k) {
      val rand = seededRandom.nextDouble()
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
  }

//  def multinomialSampling(logitValues: Array[Float], k: Int, seed: Long = 42): Array[Int] = {
////    val n = logitValues.length
////    val (distFiltered, indices) =
////      logitValues.zipWithIndex.filter { case (elem, index) => !elem.isInfinite }.sorted.unzip
//
//    val maxLogit = logitValues.max
//    val logitsExp = logitValues.map(logit => math.exp(logit - maxLogit))
//    val sumExp = logitsExp.sum
//    val probs = logitsExp.map(exp => (exp / sumExp).toFloat)
////    val probs = softmax(distFiltered)
//    val cdf = getCDF(probs)
//    val rand = new scala.util.Random(seed)
//    val samples = Array.ofDim[Int](k)
//
//    for (i <- 0 until k) {
//      val u = rand.nextDouble()
//      var j = 0
//      while (u > cdf(j)) {
//        j += 1
//      }
//      samples(i) = j
//      cdf(j) = 0.0f // remove probability mass for sampling without replacement
//    }
//    samples
//  }

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
