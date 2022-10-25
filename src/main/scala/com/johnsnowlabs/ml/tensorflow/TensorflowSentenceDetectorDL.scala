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

import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.nlp.util.io.OutputHelper
import org.apache.spark.ml.util.Identifiable
import org.tensorflow.Graph
import org.tensorflow.proto.framework.GraphDef

import scala.collection.JavaConverters._
import scala.util.Random

class TensorflowSentenceDetectorDL(
    val model: TensorflowWrapper,
    val verboseLevel: Verbose.Value = Verbose.All,
    val outputLogsPath: Option[String] = None)
    extends Serializable
    with Logging {

  private val initKey = "init"
  private val inputsKey = "inputs"
  private val targetsKey = "targets"
  private val classWeightsKey = "class_weights"
  private val dropoutKey = "dropout"
  private val learningRateKey = "learning_rate"
  private val trainingKey = "optimizer"
  private val lossKey = "loss"
  private val outputsKey = "outputs"
  private val predictionsKey = "predictions"
  private val accuracyKey = "accuracy"

  private lazy val _graphOperations = {
    val graph = new Graph()
    graph.importGraphDef(GraphDef.parseFrom(model.graph))
    graph.operations().asScala.toArray
  }

  private lazy val _inputDim: Int = {
    val op = _graphOperations.find(op => op.name() == inputsKey)
    if (op.isDefined) {
      op.get.output(0).shape().size(1).toInt
    } else {
      throw new Exception("Can't find input tensor")
    }
  }

  private lazy val _outputDim: Int = {
    val op = _graphOperations.find(op => op.name() == outputsKey)
    if (op.isDefined) {
      op.get.output(0).shape().size(1).toInt
    } else {
      throw new Exception("Can't find output tensor")
    }
  }

  def getTFModel: TensorflowWrapper = this.model

  protected def logMessage(message: String, uuid: String): Unit = {

    if (outputLogsPath.isDefined) {
      outputLog(message, uuid, true, outputLogsPath.get)
    }

  }

  def train(
      features: Array[Array[Float]],
      labels: Array[Array[Float]],
      batchSize: Int,
      epochsNumber: Int,
      learningRate: Float = 0.001f,
      validationSplit: Float = 0.0f,
      classWeights: Array[Float],
      dropout: Float = 0.0f,
      configProtoBytes: Option[Array[Byte]] = None,
      uuid: String = Identifiable.randomUID("annotator")): Unit = {

    model.createSession(configProtoBytes).runner.addTarget(initKey).run()

    val outputClassWeights = classWeights.padTo(_outputDim, 0.0f)

    val zippedDataset = features
      .map(x => x.padTo(_inputDim, 0.0f))
      .zip(labels.map(x => x.padTo(_outputDim, 0.0f)))
      .toSeq

    val allData = Random.shuffle(zippedDataset)

    val (trainDataset, validationDataset) = if (validationSplit > 0f) {
      allData.splitAt((features.length * (1 - validationSplit)).toInt)
    } else {
      // No validationSplit has been set so just use the entire training Dataset
      val emptyValid: Seq[(Array[Float], Array[Float])] = Seq((Array.empty, Array.empty))
      (allData, emptyValid)
    }

    println(f"Training $epochsNumber epochs")
    logMessage(f"Training $epochsNumber epochs", uuid)

    for (epoch <- 1 to epochsNumber) {

      var loss = 0.0f
      var acc = 0.0f
      var batches = 0

      val time = System.nanoTime()

      val randomizedTrainingData = Random.shuffle(trainDataset).toArray

      for (batch <- randomizedTrainingData.grouped(batchSize)) {

        val tensors = new TensorResources()

        val featuresArray = batch.map(x => x._1)
        val labelsArray = batch.map(x => x._2)

        val inputTensor = tensors.createTensor(featuresArray)
        val labelTensor = tensors.createTensor(labelsArray)
        val lrTensor = tensors.createTensor(learningRate)
        val classWeightsTensor = tensors.createTensor(outputClassWeights)
        val dropoutTensor = tensors.createTensor(dropout)

        val calculated = model
          .getTFSession(configProtoBytes)
          .runner
          .feed(inputsKey, inputTensor)
          .feed(targetsKey, labelTensor)
          .feed(learningRateKey, lrTensor)
          .feed(classWeightsKey, classWeightsTensor)
          .feed(dropoutKey, dropoutTensor)
          .addTarget(trainingKey)
          .fetch(lossKey)
          .fetch(accuracyKey)
          .run()
        loss += TensorResources.extractFloats(calculated.get(0))(0)
        acc += TensorResources.extractFloats(calculated.get(1))(0)
        batches += 1
        tensors.clearTensors()

      }

      acc /= batches

      if (validationSplit > 0.0) {
        val (validationFeatures, validationLabels) = validationDataset.toArray.unzip
        val (_, valid_acc) = internalPredict(
          validationFeatures,
          validationLabels,
          configProtoBytes,
          outputClassWeights)
        val endTime = (System.nanoTime() - time) / 1e9
        println(
          f"Epoch $epoch/$epochsNumber\t$endTime%.2fs\tLoss: $loss\tACC: $acc\tValidation ACC: $valid_acc")
        logMessage(
          f"Epoch $epoch/$epochsNumber\t$endTime%.2fs\tLoss: $loss\tACC: $acc\tValidation ACC: $valid_acc",
          uuid)
      } else {
        val endTime = (System.nanoTime() - time) / 1e9
        println(f"Epoch $epoch/$epochsNumber\t$endTime%.2fs\tLoss: $loss\tACC: $acc")
        logMessage(f"Epoch $epoch/$epochsNumber\t$endTime%.2fs\tLoss: $loss\tACC: $acc", uuid)
      }
    }
    println(f"Training completed.")
    logMessage(f"Training completed.", uuid)

    if (outputLogsPath.isDefined) {
      OutputHelper.exportLogFile(outputLogsPath.get)
    }
  }

  protected def internalPredict(
      features: Array[Array[Float]],
      labels: Array[Array[Float]],
      configProtoBytes: Option[Array[Byte]] = None,
      classWeights: Array[Float]): (Float, Float) = {

    val tensors = new TensorResources()

    val inputTensor = tensors.createTensor(features)
    val labelTensor = tensors.createTensor(labels)
    val classWeightsTensor = tensors.createTensor(classWeights)

    val calculated = model
      .getTFSession(configProtoBytes)
      .runner
      .feed(inputsKey, inputTensor)
      .feed(targetsKey, labelTensor)
      .feed(classWeightsKey, classWeightsTensor)
      .fetch(lossKey)
      .fetch(accuracyKey)
      .run()

    val loss = TensorResources.extractFloats(calculated.get(0))(0)
    val acc = TensorResources.extractFloats(calculated.get(1))(0)

    tensors.clearTensors()

    (loss, acc)
  }

  def predict(
      features: Array[Array[Float]],
      configProtoBytes: Option[Array[Byte]] = None): (Array[Long], Array[Float]) = {

    val tensors = new TensorResources()
    val inputTensor = tensors.createTensor(features.map(x => x.padTo(_inputDim, 0.0f)))

    val calculated = model
      .getTFSession(configProtoBytes)
      .runner
      .feed(inputsKey, inputTensor)
      .fetch(predictionsKey)
      .fetch(outputsKey)
      .run()

    val prediction = TensorResources.extractLongs(calculated.get(0))
    val outputs = TensorResources.extractFloats(calculated.get(1)).grouped(_outputDim).toArray
    val confidence = 0.until(prediction.length).map(i => outputs(i)(prediction(i).toInt)).toArray

    tensors.clearTensors()

    (prediction, confidence)
  }

}
