/*
 * Copyright 2017-2022 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.ml.tensorflow

import com.johnsnowlabs.ml.tensorflow.TensorResources._
import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.nlp.annotators.spell.context.LangModelSentence
import org.tensorflow.Graph

import scala.collection.JavaConverters._

private[johnsnowlabs] class TensorflowSpell(
    val tensorflow: TensorflowWrapper,
    val verboseLevel: Verbose.Value)
    extends Logging
    with Serializable {

  val lossKey = "Add:0"
  val dropoutRate = "dropout_rate"

  // these are the inputs to the graph
  val wordIds = "input_batch"
  val contextIds = "output_batch_cids"
  val contextWordIds = "output_batch_wids"
  val inputLens = "input_batch_lengths"

  // for fast evaluation
  val testWids = "test_wids"
  val testCids = "test_cids"
  val losses = "test_losses"

  // training stuff
  val globalStepKey = "train/global_step"
  val initialLearningRateKey = "train/initial_learning_rate"
  val finalLearningRateKey = "train/final_learning_rate"
  val updatesKey = "train/updates"

  // Controls the frequency at which we report train progress, can be disabled completely by changing the log level
  val checkPointStep = 200
  val learningRate = "train/learning_rate/Merge:0"
  val initKey = "init"

  /* returns the loss associated with the last word, given previous history  */
  def predict(
      dataset: Array[Array[Int]],
      cids: Array[Array[Int]],
      cwids: Array[Array[Int]],
      configProtoBytes: Option[Array[Byte]] = None): Iterator[Float] = {

    val tensors = new TensorResources

    val lossWords = tensorflow
      .getTFSession(configProtoBytes = configProtoBytes)
      .runner
      .feed(dropoutRate, tensors.createTensor(1.0f))
      .feed(wordIds, tensors.createTensor(dataset.map(_.dropRight(1))))
      .feed(contextIds, tensors.createTensor(cids.map(_.tail)))
      .feed(contextWordIds, tensors.createTensor(cwids.map(_.tail)))
      .fetch(lossKey)
      .run()

    tensors.clearTensors()

    val result = extractFloats(lossWords.get(0))
    val width = dataset.head.length
    result.grouped(width - 1).map(_.last)
  }

  /* returns the loss associated with the last word, given previous history  */
  def pplEachWord(
      dataset: Array[Array[Int]],
      cids: Array[Array[Int]],
      cwids: Array[Array[Int]],
      configProtoBytes: Option[Array[Byte]] = None): Array[Float] = {

    val tensors = new TensorResources

    val lossWords = tensorflow
      .getTFSession(configProtoBytes = configProtoBytes)
      .runner
      .feed(dropoutRate, tensors.createTensor(1.0f))
      .feed(wordIds, tensors.createTensor(dataset.map(_.dropRight(1))))
      .feed(contextIds, tensors.createTensor(cids.map(_.tail)))
      .feed(contextWordIds, tensors.createTensor(cwids.map(_.tail)))
      .feed(inputLens, tensors.createTensor(dataset.map(_.length)))
      .fetch(lossKey)
      .run()

    tensors.clearTensors()
    extractFloats(lossWords.get(0))
  }

  def predict_(
      dataset: Array[Array[Int]],
      cids: Array[Array[Int]],
      cwids: Array[Array[Int]],
      candCids: Array[Int],
      candWids: Array[Int],
      configProtoBytes: Option[Array[Byte]] = None) = {

    val tensors = new TensorResources
    val paths = (dataset, cids, cwids).zipped.toList

    paths.flatMap { case (pathIds, pathCids, pathWids) =>
      val lossWords = tensorflow
        .getTFSession(configProtoBytes = configProtoBytes)
        .runner
        .feed(dropoutRate, tensors.createTensor(1.0f))
        .feed(wordIds, tensors.createTensor(Array(pathIds)))
        .feed(contextIds, tensors.createTensor(Array(pathCids.tail)))
        .feed(contextWordIds, tensors.createTensor(Array(pathWids.tail)))
        .feed(testCids, tensors.createTensor(Array(candCids)))
        .feed(testWids, tensors.createTensor(Array(candWids)))
        .feed(inputLens, tensors.createTensor(Array(pathIds.length)))
        .fetch(losses)
        .run()

      tensors.clearTensors()
      val r = extractFloats(lossWords.get(0))
      r
    }
  }

  def train(
      train: => Iterator[Array[LangModelSentence]],
      valid: => Iterator[Array[LangModelSentence]],
      epochs: Int,
      batchSize: Int,
      initialRate: Float,
      finalRate: Float): Unit = {

    val graph = new Graph()
    val config = Array[Byte](50, 2, 32, 1, 56, 1)
    val session = tensorflow.createSession(Some(config))
    session.runner.addTarget(initKey).run()

    var bestScore = Double.MaxValue
    for (epoch <- 0 until epochs) {
      logger.info(s"Training language model: epoch $epoch")
      for (batch <- train) {
        val tensors = new TensorResources()
        var trainLoss = 0.0
        var trainValidWords = 0

        val tfResponse = session
          .runner()
          .fetch(lossKey)
          .fetch(globalStepKey)
          .fetch(learningRate)
          .fetch(updatesKey)
          .feed(dropoutRate, tensors.createTensor(.65f))
          .feed(wordIds, tensors.createTensor(batch.map(_.ids)))
          .feed(contextIds, tensors.createTensor(batch.map(_.cids)))
          .feed(contextWordIds, tensors.createTensor(batch.map(_.cwids)))
          .feed(inputLens, tensors.createTensor(batch.map(_.len)))
          .feed(finalLearningRateKey, tensors.createTensor(finalRate))
          .feed(initialLearningRateKey, tensors.createTensor(initialRate))
          .run()

        val loss = tfResponse.asScala.headOption match {
          case Some(e) => e
          case _ => throw new IllegalArgumentException("Error in TF loss extraction")
        }

        val gs = tfResponse.asScala.lift(1) match {
          case Some(e) => e
          case _ => throw new IllegalArgumentException("Error in TF gs extraction")
        }

        val clr = tfResponse.asScala.lift(2) match {
          case Some(e) => e
          case _ => throw new IllegalArgumentException("Error in TF clr extraction")
        }

        trainLoss += extractFloats(loss).sum
        val vws = batch.map(_.len).sum
        trainValidWords += vws

        if (extractInt(gs) % checkPointStep == 0) {
          trainLoss /= vws
          val trainPpl = math.exp(trainLoss)
          logger.debug(
            s"Training Step: ${extractInt(gs)}, LR: ${extractFloats(clr).head}\n Training PPL: $trainPpl")
          trainLoss = 0.0
          trainValidWords = 0
        }
      }

      // The end of one epoch - run validation)
      var devLoss = 0.0
      var devValidWords = 0
      val tensors = new TensorResources()

      for (batch <- valid) {
        val tfValidationResponse = session
          .runner()
          .fetch(lossKey)
          .feed(dropoutRate, tensors.createTensor(1.0f))
          .feed(wordIds, tensors.createTensor(batch.map(_.ids)))
          .feed(contextIds, tensors.createTensor(batch.map(_.cids)))
          .feed(contextWordIds, tensors.createTensor(batch.map(_.cwids)))
          .feed(inputLens, tensors.createTensor(batch.map(_.len)))
          .run()

        val validLoss = tfValidationResponse.get(0)
        devLoss += extractFloats(validLoss).sum
        devValidWords += batch.map(_.len).sum
      }
      // End of validation
      devLoss /= devValidWords
      val devPpl = math.exp(devLoss)
      logger.debug(s"Validation PPL: $devPpl")
      if (devPpl < bestScore) {
        bestScore = devPpl
      }
    }
  }
}
