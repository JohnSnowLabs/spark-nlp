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

class TensorflowSentiment(
    val tensorflow: TensorflowWrapper,
    val encoder: ClassifierDatasetEncoder,
    override val verboseLevel: Verbose.Value)
    extends Serializable
    with ClassifierMetrics {

  private val inputKey = "inputs:0"
  private val labelKey = "labels:0"
  private val learningRateKey = "lr:0"
  private val dropouttKey = "dp:0"

  private val numClasses: Int = encoder.params.tags.length

  private val predictionKey = s"softmax_output_$numClasses/Softmax:0"
  private val optimizer = s"optimizer_adam_$numClasses/Adam/Assign:0"
  private val cost = s"loss_$numClasses/softmax_cross_entropy_with_logits_sg:0"
  private val accuracy = s"accuracy_$numClasses/mean_accuracy:0"
  private val initKey = "init_all_tables"

  def train(
      inputs: (Array[Array[Float]], Array[String]),
      testInputs: Option[(Array[Array[Float]], Array[String])],
      lr: Float = 5e-3f,
      batchSize: Int = 64,
      dropout: Float = 0.5f,
      startEpoch: Int = 0,
      endEpoch: Int = 10,
      configProtoBytes: Option[Array[Byte]] = None,
      validationSplit: Float = 0.0f,
      evaluationLogExtended: Boolean = false,
      enableOutputLogs: Boolean = false,
      outputLogsPath: String,
      uuid: String = Identifiable.randomUID("sentimentdl")): Unit = {

    // Initialize
    if (startEpoch == 0)
      tensorflow
        .createSession(configProtoBytes = configProtoBytes)
        .runner
        .addTarget(initKey)
        .run()

    val (trainSet, validationSet, testSet) = buildDatasets(inputs, testInputs, validationSplit)

    println(
      s"Training started - epochs: $endEpoch - learning_rate: $lr - batch_size: $batchSize - training_examples: ${trainSet.length}")
    outputLog(
      s"Training started - epochs: $endEpoch - learning_rate: $lr - batch_size: $batchSize - training_examples: ${trainSet.length}",
      uuid,
      enableOutputLogs,
      outputLogsPath)

    for (epoch <- startEpoch until endEpoch) {

      val time = System.nanoTime()
      var batches = 0
      var loss = 0f
      var acc = 0f
      val learningRate = lr / (1 + dropout * epoch)

      for (batch <- trainSet.grouped(batchSize)) {
        val tensors = new TensorResources()

        val inputArrays = batch.map(x => x._1)
        val labelsArray = batch.map(x => x._2)

        val inputTensor = tensors.createTensor(inputArrays)
        val labelTensor = tensors.createTensor(labelsArray)
        val lrTensor = tensors.createTensor(learningRate)
        val dpTensor = tensors.createTensor(dropout)

        val calculated = tensorflow
          .getTFSession(configProtoBytes = configProtoBytes)
          .runner
          .feed(inputKey, inputTensor)
          .feed(labelKey, labelTensor)
          .feed(learningRateKey, lrTensor)
          .feed(dropouttKey, dpTensor)
          .fetch(predictionKey)
          .fetch(optimizer)
          .fetch(cost)
          .fetch(accuracy)
          .run()

        loss += TensorResources.extractFloats(calculated.get(2))(0)
        acc += TensorResources.extractFloats(calculated.get(3))(0)
        batches += 1

        tensors.clearTensors()
      }
      acc /= (trainSet.length / batchSize)
      acc = acc.min(1.0f).max(0.0f)

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
          s"Quality on validation dataset (${validationSplit * 100}%), validation examples = ${validationSet.length}")
        outputLog(
          s"Quality on validation dataset (${validationSplit * 100}%), validation examples = ${validationSet.length}",
          uuid,
          enableOutputLogs,
          outputLogsPath)

        measure(validationSet, extended = evaluationLogExtended, enableOutputLogs, outputLogsPath)
      }

      if (testSet.nonEmpty) {
        println(s"Quality on test dataset: ")
        outputLog("Quality on test dataset: ", uuid, enableOutputLogs, outputLogsPath)

        measure(testSet, extended = evaluationLogExtended, enableOutputLogs, outputLogsPath)
      }

    }

    if (enableOutputLogs) {
      OutputHelper.exportLogFile(outputLogsPath)
    }
  }

  private def buildDatasets(
      inputs: (Array[Array[Float]], Array[String]),
      testInputs: Option[(Array[Array[Float]], Array[String])],
      validationSplit: Float): (
      Array[(Array[Float], Array[Int])],
      Array[(Array[Float], Array[Int])],
      Array[(Array[Float], Array[Int])]) = {

    val trainingDataset = Random.shuffle(encodeInputs(inputs).toSeq).toArray
    val testDataset: Array[(Array[Float], Array[Int])] =
      if (testInputs.isDefined) encodeInputs(testInputs.get) else Array.empty

    val sample: Int = (trainingDataset.length * validationSplit).toInt

    val (newTrainDataset, validateDatasetSample) = if (validationSplit > 0f) {
      val (trainingSample, trainingSet) = trainingDataset.splitAt(sample)
      (trainingSet, trainingSample)
    } else {
      // No validationSplit has been set so just use the entire training Dataset
      val emptyValid: Array[(Array[Float], Array[Int])] = Array((Array.empty, Array.empty))
      (trainingDataset, emptyValid)
    }

    (newTrainDataset, validateDatasetSample, testDataset)
  }

  private def encodeInputs(
      inputs: (Array[Array[Float]], Array[String])): Array[(Array[Float], Array[Int])] = {

    val (embeddings, labels) = inputs
    val encodedLabels = encoder.encodeTags(labels)

    embeddings.zip(encodedLabels)
  }

  def predict(
      docs: Seq[(Int, Seq[Annotation])],
      configProtoBytes: Option[Array[Byte]] = None,
      threshold: Float = 0.6f,
      thresholdLabel: String = "neutral"): Seq[Annotation] = {

    val tensors = new TensorResources()

    // FixMe: implement batchSize

    val inputs = encoder.extractSentenceEmbeddings(docs)

    val calculated = tensorflow
      .getTFSession(configProtoBytes = configProtoBytes)
      .runner
      .feed(inputKey, tensors.createTensor(inputs))
      .fetch(predictionKey)
      .run()

    val tagsId = TensorResources.extractFloats(calculated.get(0)).grouped(numClasses).toArray
    val tagsName = encoder.decodeOutputData(tagIds = tagsId)
    tensors.clearTensors()

    docs.flatMap { sentence =>
      sentence._2.zip(tagsName).map { case (content, score) =>
        val label = score.find(_._1 == score.maxBy(_._2)._1).map(_._1).getOrElse("NA")
        val confidenceScore = score.find(_._1 == score.maxBy(_._2)._1).map(_._2).getOrElse(0.0f)
        val finalLabel = if (confidenceScore >= threshold) label else thresholdLabel

        Annotation(
          annotatorType = AnnotatorType.CATEGORY,
          begin = content.begin,
          end = content.end,
          result = finalLabel,
          metadata = Map("sentence" -> sentence._1.toString) ++ score.flatMap(x =>
            Map(x._1 -> x._2.toString)))
      }
    }

  }

  def internalPredict(
      inputs: Array[Array[Float]],
      configProtoBytes: Option[Array[Byte]] = None): Array[Int] = {

    val tensors = new TensorResources()

    val calculated = tensorflow
      .getTFSession(configProtoBytes = configProtoBytes)
      .runner
      .feed(inputKey, tensors.createTensor(inputs))
      .fetch(predictionKey)
      .run()

    val tagsId = TensorResources.extractFloats(calculated.get(0)).grouped(numClasses).toArray
    val predictedLabels = tagsId.map { score =>
      val labelId = score.zipWithIndex.maxBy(_._1)._2
      labelId
    }
    tensors.clearTensors()
    predictedLabels
  }

  def measure(
      labeled: Array[(Array[Float], Array[Int])],
      extended: Boolean = true,
      enableOutputLogs: Boolean = false,
      outputLogsPath: String,
      batchSize: Int = 100): (Float, Float) = {

    val started = System.nanoTime()

    // ToDo: Add batch strategy

    val predicted = mutable.Map[Int, Int]()
    val correct = mutable.Map[Int, Int]()

    val truePositives = mutable.Map[Int, Int]()
    val falsePositives = mutable.Map[Int, Int]()
    val falseNegatives = mutable.Map[Int, Int]()

    val originalEmbeddings = labeled.map(x => x._1)
    val originalLabels = labeled.map(x => x._2).map { x =>
      x.zipWithIndex.maxBy(_._1)._2
    }

    val predictedLabels = internalPredict(originalEmbeddings)
    val labeledPredicted = predictedLabels.zip(originalLabels)

    for (label <- labeledPredicted) {
      val predict = label._1
      val original = label._2

      correct(original) = correct.getOrElse(original, 0) + 1
      predicted(predict) = predicted.getOrElse(predict, 0) + 1

      if (original == predict) {
        truePositives(original) = truePositives.getOrElse(original, 0) + 1
      } else {
        falsePositives(predict) = falsePositives.getOrElse(predict, 0) + 1
        falseNegatives(original) = falseNegatives.getOrElse(original, 0) + 1
      }
    }

    val endTime = (System.nanoTime() - started) / 1e9
    println(f"time to finish evaluation: $endTime%.2fs")

    val labels = (correct.keys ++ predicted.keys).toSeq.distinct

    aggregatedMetrics(
      labels.map(_.toString),
      truePositives.map(tp => (tp._1.toString, tp._2)).toMap,
      falsePositives.map(fp => (fp._1.toString, fp._2)).toMap,
      falseNegatives.map(fn => (fn._1.toString, fn._2)).toMap,
      extended,
      enableOutputLogs,
      outputLogsPath)

  }

}
