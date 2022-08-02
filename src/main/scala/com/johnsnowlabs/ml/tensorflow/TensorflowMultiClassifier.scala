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
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import org.apache.spark.ml.util.Identifiable

import scala.util.Random

class TensorflowMultiClassifier(
    val tensorflow: TensorflowWrapper,
    val encoder: ClassifierDatasetEncoder,
    override val verboseLevel: Verbose.Value)
    extends Serializable
    with Logging {

  private val inputKey = "inputs:0"
  private val labelKey = "labels:0"
  private val sequenceLengthKey = "sequence_length:0"
  private val learningRateKey = "lr:0"

  private val numClasses: Int = encoder.params.tags.length

  private val predictionKey = s"sigmoid_output_$numClasses/Sigmoid:0"
  private val optimizerKey = s"optimizer_adam_$numClasses/Adam"
  private val lossKey = s"loss_$numClasses/bce_loss/weighted_loss/value:0"
  private val accuracyKey = s"accuracy_$numClasses/mean_accuracy:0"
  private val metricsF1 = s"metrics_$numClasses/f1/f1_score:0"
  private val metricsAccKey = s"metrics_$numClasses/accuracy/mean_accuracy:0"
  private val metricsLossKey = s"metrics_$numClasses/loss/bce_loss/weighted_loss/value:0"
  private val metricsTPRKey = s"metrics_$numClasses/f1/truediv_4:0"

  private val initKey = "init_all_tables"

  def reshapeInputFeatures(batch: Array[Array[Array[Float]]]): Array[Array[Array[Float]]] = {
    val sequencesLength = batch.map(x => x.length)
    val maxSentenceLength = sequencesLength.max
    val dimension = batch(0).head.length
    batch.map { sentence =>
      if (sentence.length >= maxSentenceLength) {
        sentence.take(maxSentenceLength)
      } else {
        val diff = maxSentenceLength - sentence.length
        sentence ++ Array.fill(diff)(Array.fill(dimension)(0.0f))
      }
    }
  }

  def train(
      inputs: Array[Array[Array[Float]]],
      labels: Array[Array[String]],
      classNum: Int,
      lr: Float = 5e-3f,
      batchSize: Int = 64,
      startEpoch: Int = 0,
      endEpoch: Int = 10,
      configProtoBytes: Option[Array[Byte]] = None,
      validationSplit: Float = 0.0f,
      shuffleEpoch: Boolean = false,
      enableOutputLogs: Boolean = false,
      outputLogsPath: String,
      uuid: String = Identifiable.randomUID("multiclassifierdl")): Unit = {

    // Initialize
    if (startEpoch == 0)
      tensorflow
        .createSession(configProtoBytes = configProtoBytes)
        .runner
        .addTarget(initKey)
        .run()

    val encodedLabels = encoder.encodeTagsMultiLabel(labels)
    val zippedInputsLabels = inputs.zip(encodedLabels).toSeq
    val trainingDataset = Random.shuffle(zippedInputsLabels)

    val sample: Int = (trainingDataset.length * validationSplit).toInt

    val (trainDatasetSeq, validateDatasetSample) = if (validationSplit > 0f) {
      val (trainingSample, trainingSet) = trainingDataset.splitAt(sample)
      (trainingSet.toArray, trainingSample.toArray)
    } else {
      // No validationSplit has been set so just use the entire training Dataset
      val emptyValid: Seq[(Array[Array[Float]], Array[Float])] = Seq((Array.empty, Array.empty))
      (trainingDataset.toArray, emptyValid.toArray)
    }

    println(
      s"Training started - epochs: $endEpoch - learning_rate: $lr - batch_size: $batchSize - training_examples: ${trainDatasetSeq.length} - classes: $classNum")
    outputLog(
      s"Training started - epochs: $endEpoch - learning_rate: $lr - batch_size: $batchSize - training_examples: ${trainDatasetSeq.length} - classes: $classNum",
      uuid,
      enableOutputLogs,
      outputLogsPath)

    for (epoch <- startEpoch until endEpoch) {

      val time = System.nanoTime()
      var batches = 0
      var loss = 0f
      var acc = 0f
      val learningRate = lr / (1 + 0.2 * epoch)

      val shuffledBatch = if (shuffleEpoch) {
        Random.shuffle(trainDatasetSeq.toSeq).toArray
      } else trainDatasetSeq

      for (batch <- shuffledBatch.grouped(batchSize)) {
        val tensors = new TensorResources()

        val sequenceLengthArrays = batch.map(x => x._1.length)
        val inputArrays = reshapeInputFeatures(batch.map(x => x._1))
        val labelsArray = batch.map(x => x._2)

        val inputTensor = tensors.createTensor(inputArrays)
        val labelTensor = tensors.createTensor(labelsArray)
        val sequenceLengthTensor = tensors.createTensor(sequenceLengthArrays)
        val lrTensor = tensors.createTensor(learningRate.toFloat)

        val calculated = tensorflow
          .getTFSession(configProtoBytes = configProtoBytes)
          .runner
          .feed(inputKey, inputTensor)
          .feed(labelKey, labelTensor)
          .feed(sequenceLengthKey, sequenceLengthTensor)
          .feed(learningRateKey, lrTensor)
          .fetch(predictionKey)
          .fetch(lossKey)
          .fetch(accuracyKey)
          .addTarget(optimizerKey)
          .run()

        loss += TensorResources.extractFloats(calculated.get(1))(0)
        acc += TensorResources.extractFloats(calculated.get(2))(0)
        batches += 1

        tensors.clearTensors()

      }
      acc /= (trainDatasetSeq.length / batchSize)
      loss /= (trainDatasetSeq.length / batchSize)

      if (validationSplit > 0.0) {
        val validationAccuracy = measure(validateDatasetSample)
        val endTime = (System.nanoTime() - time) / 1e9
        println(
          f"Epoch ${epoch + 1}/$endEpoch - $endTime%.2fs - loss: $loss - acc: $acc - val_loss: ${validationAccuracy(
              0)} - val_acc: ${validationAccuracy(1)} - val_f1: ${validationAccuracy(
              2)} - val_tpr: ${validationAccuracy(3)} - batches: $batches")
        outputLog(
          f"Epoch $epoch/$endEpoch - $endTime%.2fs - loss: $loss - acc: $acc - val_loss: ${validationAccuracy(
              0)} - val_acc: ${validationAccuracy(1)} - val_f1: ${validationAccuracy(
              2)} - val_tpr: ${validationAccuracy(3)} - batches: $batches",
          uuid,
          enableOutputLogs,
          outputLogsPath)
      } else {
        val endTime = (System.nanoTime() - time) / 1e9
        println(f"Epoch ${epoch + 1}/$endEpoch - $endTime%.2fs - loss: $loss - batches: $batches")
        outputLog(
          f"Epoch $epoch/$endEpoch - $endTime%.2fs - loss: $loss - batches: $batches",
          uuid,
          enableOutputLogs,
          outputLogsPath)
      }

    }

    if (enableOutputLogs) {
      OutputHelper.exportLogFile(outputLogsPath)
    }
  }

  def predict(
      docs: Seq[(Int, Seq[Annotation])],
      threshold: Float = 0.5f,
      configProtoBytes: Option[Array[Byte]] = None): Seq[Annotation] = {

    val tensors = new TensorResources()

    val inputs = encoder.extractSentenceEmbeddingsMultiLabelPredict(docs)

    val sequenceLengthArrays = inputs.map(x => x.length)
    val inputsReshaped = reshapeInputFeatures(inputs)

    val calculated = tensorflow
      .getTFSession(configProtoBytes = configProtoBytes)
      .runner
      .feed(inputKey, tensors.createTensor(inputsReshaped))
      .feed(sequenceLengthKey, tensors.createTensor(sequenceLengthArrays))
      .fetch(predictionKey)
      .run()

    val tagsId = TensorResources.extractFloats(calculated.get(0)).grouped(numClasses).toArray
    val tagsName = encoder.decodeOutputData(tagIds = tagsId)
    tensors.clearTensors()

    tagsName.flatMap { score =>
      val labels = score.filter(x => x._2 >= threshold).map(x => x._1)
      val documentBegin = docs.head._2.head.begin
      val documentEnd = docs.last._2.last.end
      labels.map { label =>
        Annotation(
          annotatorType = AnnotatorType.CATEGORY,
          begin = documentBegin,
          end = documentEnd,
          result = label,
          metadata = Map("sentence" -> "0") ++ score.flatMap(x => Map(x._1 -> x._2.toString)))
      }
    }
  }

  def internalPredict(
      inputs: Array[Array[Array[Float]]],
      labels: Array[Array[Float]],
      configProtoBytes: Option[Array[Byte]] = None): Array[Float] = {

    val tensors = new TensorResources()

    val sequenceLengthArrays = inputs.map(x => x.length)
    val inputsReshaped = reshapeInputFeatures(inputs)

    val calculated = tensorflow
      .getTFSession(configProtoBytes = configProtoBytes)
      .runner
      .feed(inputKey, tensors.createTensor(inputsReshaped))
      .feed(labelKey, tensors.createTensor(labels))
      .feed(sequenceLengthKey, tensors.createTensor(sequenceLengthArrays))
      .fetch(metricsLossKey)
      .fetch(metricsAccKey)
      .fetch(metricsF1)
      .fetch(metricsTPRKey)
      .run()

    val valLoss = TensorResources.extractFloats(calculated.get(0))(0)
    val valAcc = TensorResources.extractFloats(calculated.get(1))(0)
    val valF1 = TensorResources.extractFloats(calculated.get(2))(0)
    val valTPR = TensorResources.extractFloats(calculated.get(3))(0)

    tensors.clearTensors()
    Array(valLoss, valAcc, valF1, valTPR)

  }

  def measure(
      labeled: Array[(Array[Array[Float]], Array[Float])],
      batchSize: Int = 100): Array[Float] = {

    var loss = 0f
    var acc = 0f
    var f1 = 0f
    var tpr = 0f

    for (batch <- labeled.grouped(batchSize)) {
      val originalEmbeddings = batch.map(x => x._1)
      val originalLabels = batch.map(x => x._2)

      val metricsArray = internalPredict(originalEmbeddings, originalLabels)
      loss += metricsArray(0)
      acc += metricsArray(1)
      f1 += metricsArray(2)
      tpr += metricsArray(3)
    }

    val avgSize = labeled.grouped(batchSize).length
    loss /= avgSize
    acc /= avgSize
    f1 /= avgSize
    tpr /= avgSize

    Array(loss, acc, f1, tpr)

  }

}
