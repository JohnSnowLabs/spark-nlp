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

import com.johnsnowlabs.nlp.annotators.classifier.dl.ClassifierMetrics
import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.nlp.util.io.OutputHelper
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import org.apache.spark.ml.util.Identifiable

import scala.collection.mutable
import scala.util.Random

class TensorflowMultiClassifier(
    val tensorflow: TensorflowWrapper,
    val encoder: ClassifierDatasetEncoder,
    val testEncoder: Option[ClassifierDatasetEncoder],
    override val verboseLevel: Verbose.Value)
    extends Serializable
    with ClassifierMetrics {

  private val inputKey = "inputs:0"
  private val labelKey = "labels:0"
  private val sequenceLengthKey = "sequence_length:0"
  private val learningRateKey = "lr:0"

  private val numClasses: Int = encoder.params.tags.length

  private val predictionKey = s"sigmoid_output_$numClasses/Sigmoid:0"
  private val optimizerKey = s"optimizer_adam_$numClasses/Adam"
  private val lossKey = s"loss_$numClasses/bce_loss/weighted_loss/value:0"
  private val accuracyKey = s"accuracy_$numClasses/mean_accuracy:0"

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
      trainInputs: (Array[Array[Array[Float]]], Array[Array[String]]),
      testInputs: Option[(Array[Array[Array[Float]]], Array[Array[String]])],
      classNum: Int,
      lr: Float = 5e-3f,
      batchSize: Int = 64,
      startEpoch: Int = 0,
      endEpoch: Int = 10,
      configProtoBytes: Option[Array[Byte]] = None,
      validationSplit: Float = 0.0f,
      evaluationLogExtended: Boolean = false,
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

    val (trainSet, validationSet, testSet) =
      buildDatasets(trainInputs, testInputs, validationSplit)

    println(
      s"Training started - epochs: $endEpoch - learning_rate: $lr - batch_size: $batchSize - training_examples: ${trainSet.length} - classes: $classNum")
    outputLog(
      s"Training started - epochs: $endEpoch - learning_rate: $lr - batch_size: $batchSize - training_examples: ${trainSet.length} - classes: $classNum",
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
        Random.shuffle(trainSet.toSeq).toArray
      } else trainSet

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
      acc /= (trainSet.length / batchSize)
      loss /= (trainSet.length / batchSize)

      val endTime = (System.nanoTime() - time) / 1e9
      println(
        f"Epoch ${epoch + 1}/$endEpoch - $endTime%.2fs - loss: $loss - acc: $acc - batches: $batches")
      outputLog(
        f"Epoch $epoch/$endEpoch - $endTime%.2fs - loss: $loss - acc: $acc - batches: $batches",
        uuid,
        enableOutputLogs,
        outputLogsPath)

      if (validationSplit > 0.0) {
        println(
          s"Quality on validation dataset (${validationSplit * 100}%), validation examples = ${validationSet.length} ")
        outputLog(
          s"Quality on validation dataset (${validationSplit * 100}%), validation examples = ${validationSet.length} ",
          uuid,
          enableOutputLogs,
          outputLogsPath)

        measure(
          validationSet,
          "validation",
          extended = evaluationLogExtended,
          enableOutputLogs,
          outputLogsPath)
      }

      if (testInputs.isDefined) {
        println(s"Quality on test dataset: ")
        outputLog(s"Quality on test dataset: ", uuid, enableOutputLogs, outputLogsPath)

        measure(
          testSet,
          "test",
          extended = evaluationLogExtended,
          enableOutputLogs,
          outputLogsPath)
      }

    }

    if (enableOutputLogs) {
      OutputHelper.exportLogFile(outputLogsPath)
    }
  }

  private def buildDatasets(
      inputs: (Array[Array[Array[Float]]], Array[Array[String]]),
      testInputs: Option[(Array[Array[Array[Float]]], Array[Array[String]])],
      validationSplit: Float): (
      Array[(Array[Array[Float]], Array[Float])],
      Array[(Array[Array[Float]], Array[Float])],
      Array[(Array[Array[Float]], Array[Float])]) = {

    val trainingDataset = Random.shuffle(encodeInputs(inputs, "train").toSeq)
    val sample: Int = (trainingDataset.length * validationSplit).toInt

    val (newTrainDataset, validateDatasetSample) = if (validationSplit > 0f) {
      val (trainingSample, trainingSet) = trainingDataset.splitAt(sample)
      (trainingSet.toArray, trainingSample.toArray)
    } else {
      // No validationSplit has been set so just use the entire training Dataset
      val emptyValid: Seq[(Array[Array[Float]], Array[Float])] = Seq((Array.empty, Array.empty))
      (trainingDataset.toArray, emptyValid.toArray)
    }

    val testDataset: Array[(Array[Array[Float]], Array[Float])] =
      if (testInputs.isDefined) encodeInputs(testInputs.get, "test") else Array.empty

    (newTrainDataset, validateDatasetSample, testDataset)
  }

  private def encodeInputs(
      inputs: (Array[Array[Array[Float]]], Array[Array[String]]),
      sourceData: String): Array[(Array[Array[Float]], Array[Float])] = {

    val (embeddings, labels) = inputs
    val myEncoder = if (sourceData == "train") encoder else testEncoder.get
    val encodedLabels = myEncoder.encodeTagsMultiLabel(labels)

    embeddings.zip(encodedLabels)
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
    val tagsWithScoresBatch = encoder.decodeOutputData(tagIds = tagsId)
    tensors.clearTensors()

    tagsWithScoresBatch.flatMap { tagsWithScores =>
      val labels = tagsWithScores.filter(tagWithScore => tagWithScore._2 >= threshold).map(_._1)
      val documentBegin = docs.head._2.head.begin
      val documentEnd = docs.last._2.last.end
      labels.map { label =>
        Annotation(
          annotatorType = AnnotatorType.CATEGORY,
          begin = documentBegin,
          end = documentEnd,
          result = label,
          metadata = Map("sentence" -> "0") ++ tagsWithScores.flatMap(tagWithScore =>
            Map(tagWithScore._1 -> tagWithScore._2.toString)))
      }
    }
  }

  def measure(
      inputs: Array[(Array[Array[Float]], Array[Float])],
      sourceData: String,
      extended: Boolean = false,
      enableOutputLogs: Boolean = false,
      outputLogsPath: String,
      batchSize: Int = 100,
      uuid: String = Identifiable.randomUID("annotator")): (Float, Float) = {

    val started = System.nanoTime()

    val evaluationEncoder = if (sourceData == "validation") encoder else testEncoder.get

    val truePositives = mutable.Map[String, Int]()
    val falsePositives = mutable.Map[String, Int]()
    val falseNegatives = mutable.Map[String, Int]()
    val labels = mutable.Map[String, Int]()

    for (batch <- inputs.grouped(batchSize)) {
      val originalEmbeddings = batch.map(x => x._1)
      val originalLabels = batch.map(x => x._2)
      val tagsWithScoresBatch = evaluationEncoder.decodeOutputData(tagIds = originalLabels)
      val groundTruthLabels = tagsWithScoresBatch.map { tagsWithScores =>
        tagsWithScores.map(tagWithScore => if (tagWithScore._2 >= 1.0) 1 else 0)
      }

      val predictedLabels = internalPredict(originalEmbeddings)
      groundTruthLabels.zip(predictedLabels).foreach { encodedLabels =>
        val encodedGroundTruth = encodedLabels._1
        val encodedPrediction = encodedLabels._2

        encodedGroundTruth.zipWithIndex.foreach { case (groundTruth, index) =>
          val prediction = encodedPrediction(index)

          val label = evaluationEncoder.tags(index)
          labels(label) = labels.getOrElse(label, 0) + 1

          if (groundTruth == 1 && prediction == 1) {
            truePositives(label) = truePositives.getOrElse(label, 0) + 1
          }

          if (groundTruth == 1 && prediction == 0) {
            falseNegatives(label) = falseNegatives.getOrElse(label, 0) + 1
          }

          if (groundTruth == 0 && prediction == 1) {
            falsePositives(label) = falsePositives.getOrElse(label, 0) + 1
          }
        }

      }

    }

    val endTime = (System.nanoTime() - started) / 1e9
    println(f"time to finish evaluation: $endTime%.2fs")

    aggregatedMetrics(
      labels.keys.toSeq,
      truePositives.toMap,
      falsePositives.toMap,
      falseNegatives.toMap,
      extended,
      enableOutputLogs,
      outputLogsPath)

  }

  def internalPredict(
      inputs: Array[Array[Array[Float]]],
      configProtoBytes: Option[Array[Byte]] = None,
      threshold: Float = 0.5f): Array[Array[Int]] = {

    val tensors = new TensorResources()
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
    val tagsWithScoresBatch = encoder.decodeOutputData(tagIds = tagsId)
    tensors.clearTensors()

    tagsWithScoresBatch.map { tagsWithScores =>
      tagsWithScores.map(tagWithScore => if (tagWithScore._2 >= threshold) 1 else 0)
    }
  }

}
