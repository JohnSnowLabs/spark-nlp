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

import com.johnsnowlabs.ml.crf.TextSentenceLabels
import com.johnsnowlabs.nlp.annotators.common.WordpieceEmbeddingsSentence
import com.johnsnowlabs.nlp.annotators.ner.{ModelMetrics, Verbose}
import com.johnsnowlabs.nlp.util.io.OutputHelper
import org.apache.spark.ml.util.Identifiable
import org.tensorflow.Session

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import scala.collection.{Map, mutable}

class TensorflowNer(
    val tensorflow: TensorflowWrapper,
    val encoder: NerDatasetEncoder,
    override val verboseLevel: Verbose.Value)
    extends Serializable
    with Logging {

  override def getLogName: String = "NerDL"

  private val charIdsKey = "char_repr/char_ids:0"
  private val wordLengthsKey = "char_repr/word_lengths:0"
  private val wordEmbeddingsKey = "word_repr_1/word_embeddings:0"
  private val sentenceLengthsKey = "word_repr/sentence_lengths:0"
  private val dropoutKey = "training/dropout:0"

  private val learningRateKey = "training/lr:0"
  private val labelsKey = "training/labels:0"

  private val lossKey = "inference/Mean:0"
  private val scoresKey = "context_repr/scores:0"
  private val trainingKey = "training_1/Adam"
  private val predictKey = "inference/cond_2/Merge:0"

  private val initKey = "training_1/init"

  def predict(
      dataset: Array[WordpieceEmbeddingsSentence],
      configProtoBytes: Option[Array[Byte]],
      includeConfidence: Boolean,
      includeAllConfidenceScores: Boolean,
      batchSize: Int): Array[Array[(String, Option[Array[Map[String, String]]])]] = {

    val result = ArrayBuffer[Array[(String, Option[Array[Map[String, String]]])]]()

    for (batch <- dataset.grouped(batchSize); if batch.length > 0) {
      val batchInput = encoder.encodeInputData(batch)

      if (batchInput.sentenceLengths.length == 0)
        for (_ <- batch) {
          result.append(Array.empty[(String, Option[Array[Map[String, String]]])])
        }
      else {
        val tensors = new TensorResources()

        val calculator = tensorflow
          .getTFSession(configProtoBytes = configProtoBytes)
          .runner
          .feed(sentenceLengthsKey, tensors.createTensor(batchInput.sentenceLengths))
          .feed(wordEmbeddingsKey, tensors.createTensor(batchInput.wordEmbeddings))
          .feed(wordLengthsKey, tensors.createTensor(batchInput.wordLengths))
          .feed(charIdsKey, tensors.createTensor(batchInput.charIds))
          .feed(dropoutKey, tensors.createTensor(1.0f))
          .fetch(predictKey)

        val calculatorInc = if (includeConfidence) calculator.fetch(scoresKey) else calculator

        val calculated = calculatorInc.run().asScala

        val allTags = encoder.tags
        val tagIds = TensorResources.extractInts(calculated.head)

        val confidence: Option[Seq[Array[Float]]] = {
          try {
            if (includeConfidence) {
              val scores = TensorResources.extractFloats(calculated(1))
              require(
                scores.length % tagIds.length == 0,
                "tag size mismatch against feed size. please report an issue.")
              val exp =
                scores.map(s => math.exp(s.toDouble)).grouped(scores.length / tagIds.length).toSeq
              val probs = if (includeAllConfidenceScores) {
                // Only include the score of the predicted tag
                exp.map(d =>
                  d.map(_ / d.sum)
                    .map(p =>
                      try {
                        BigDecimal(p).setScale(4, BigDecimal.RoundingMode.HALF_UP).toFloat
                      } catch {
                        case _: Exception => 0.0f
                      }))
              } else {
                exp.map(d =>
                  try {
                    Array(
                      BigDecimal(d.map(_ / d.sum).max)
                        .setScale(4, BigDecimal.RoundingMode.HALF_UP)
                        .toFloat)
                  } catch {
                    case _: Exception => Array(0.0f)
                  })
              }
              Some(probs)
            } else {
              None
            }
          } catch {
            case _: Exception =>
              Option(Seq.fill(tagIds.length)(Array.fill(allTags.length)(0.0f)))
          }
        }

        val predictedTags = encoder.decodeOutputData(tagIds)

        val sentenceTags = encoder.convertBatchTags(
          predictedTags,
          allTags,
          batchInput.sentenceLengths,
          confidence,
          includeAllConfidenceScores = includeAllConfidenceScores)

        calculated.foreach(_.close())
        tensors.clearSession(calculated)
        tensors.clearTensors()

        result.appendAll(sentenceTags)
      }
    }

    result.toArray
  }

  def getPiecesTags(
      tokenTags: TextSentenceLabels,
      sentence: WordpieceEmbeddingsSentence): Array[String] = {
    var i = -1

    sentence.tokens.map { _ =>
      i += 1
      tokenTags.labels(i)
    }
  }

  def getPiecesTags(
      tokenTags: Array[TextSentenceLabels],
      sentences: Array[WordpieceEmbeddingsSentence]): Array[Array[String]] = {
    tokenTags.zip(sentences).map { case (tags, sentence) =>
      getPiecesTags(tags, sentence)
    }
  }

  def train(
      trainDataset: => Iterator[Array[(TextSentenceLabels, WordpieceEmbeddingsSentence)]],
      trainLength: Long,
      validDataset: => Iterator[Array[(TextSentenceLabels, WordpieceEmbeddingsSentence)]],
      validLength: Long,
      lr: Float,
      po: Float,
      dropout: Float,
      batchSize: Int = 8,
      useBestModel: Boolean = false,
      bestModelMetricPreference: String = ModelMetrics.microF1,
      startEpoch: Int = 0,
      endEpoch: Int,
      graphFileName: String = "",
      test: => Iterator[Array[(TextSentenceLabels, WordpieceEmbeddingsSentence)]] =
        Iterator.empty,
      configProtoBytes: Option[Array[Byte]] = None,
      validationSplit: Float = 0.0f,
      evaluationLogExtended: Boolean = false,
      includeConfidence: Boolean = false,
      enableOutputLogs: Boolean = false,
      outputLogsPath: String,
      uuid: String = Identifiable.randomUID("annotator")): Session = {

    var bestModelMetric: String = ModelMetrics.loss
    var lastTestMircoF1, lastValMicroF1: Float = 0.0f
    var lastLoss: Float = Float.MaxValue

    if (test.nonEmpty) {
      bestModelMetric =
        if (bestModelMetricPreference == ModelMetrics.microF1) ModelMetrics.testMicroF1
        else ModelMetrics.testMacroF1
    } else if (validationSplit > 0.0) {
      bestModelMetric =
        if (bestModelMetricPreference == ModelMetrics.microF1) ModelMetrics.valMicroF1
        else ModelMetrics.valMacroF1
    } else {
      bestModelMetric = ModelMetrics.loss
    }

    log(s"Name of the selected graph: $graphFileName", Verbose.Epochs)
    outputLog(
      s"Name of the selected graph: $graphFileName",
      uuid,
      enableOutputLogs,
      outputLogsPath)

    var lastCheckPoints: Session = tensorflow.createSession(configProtoBytes = configProtoBytes)

    // Initialize
    if (startEpoch == 0)
      lastCheckPoints.runner.addTarget(initKey).run()

    println(
      s"Training started - total epochs: $endEpoch - lr: $lr - batch size: $batchSize - labels: ${encoder.tags.length} " +
        s"- chars: ${encoder.chars.length} - training examples: $trainLength")

    outputLog(
      s"Training started - total epochs: $endEpoch - lr: $lr - batch size: $batchSize - labels: ${encoder.tags.length} " +
        s"- chars: ${encoder.chars.length} - training examples: $trainLength",
      uuid,
      enableOutputLogs,
      outputLogsPath)

    // Train
    for (epoch <- startEpoch until endEpoch) {

      val learningRate = lr / (1 + po * epoch)

      println(
        s"Epoch ${epoch + 1}/$endEpoch started, lr: $learningRate, dataset size: $trainLength")
      outputLog("\n", uuid, enableOutputLogs, outputLogsPath)
      outputLog(
        s"Epoch ${epoch + 1}/$endEpoch started, lr: $learningRate, dataset size: $trainLength",
        uuid,
        enableOutputLogs,
        outputLogsPath)

      val time = System.nanoTime()
      var batches = 0
      var loss = 0f
      for (batch <- trainDataset) {
        val sentences = batch.map(r => r._2)
        val tags = getPiecesTags(batch.map(r => r._1), sentences)

        val batchInput = encoder.encodeInputData(sentences)
        val batchTags = encoder.encodeTags(tags)

        val tensors = new TensorResources()
        val calculated = tensorflow
          .getTFSession(configProtoBytes = configProtoBytes)
          .runner
          .feed(sentenceLengthsKey, tensors.createTensor(batchInput.sentenceLengths))
          .feed(wordEmbeddingsKey, tensors.createTensor(batchInput.wordEmbeddings))
          .feed(wordLengthsKey, tensors.createTensor(batchInput.wordLengths))
          .feed(charIdsKey, tensors.createTensor(batchInput.charIds))
          .feed(labelsKey, tensors.createTensor(batchTags))
          .feed(dropoutKey, tensors.createTensor(dropout))
          .feed(learningRateKey, tensors.createTensor(learningRate))
          .fetch(lossKey)
          .addTarget(trainingKey)
          .run()
          .asScala

        loss += TensorResources.extractFloats(calculated.head).head

        calculated.foreach(_.close())
        tensors.clearSession(calculated)
        tensors.clearTensors()
        batches += 1
      }

      val endTime = (System.nanoTime() - time) / 1e9
      println(f"Epoch ${epoch + 1}/$endEpoch - $endTime%.2fs - loss: $loss - batches: $batches")
      outputLog("\n", uuid, enableOutputLogs, outputLogsPath)
      outputLog(
        f"Epoch ${epoch + 1}/$endEpoch - $endTime%.2fs - loss: $loss - batches: $batches",
        uuid,
        enableOutputLogs,
        outputLogsPath)

      if (validationSplit > 0.0) {
        println(
          s"Quality on validation dataset (${validationSplit * 100}%), validation examples = $validLength")
        outputLog(
          s"Quality on validation dataset (${validationSplit * 100}%), validation examples = $validLength",
          uuid,
          enableOutputLogs,
          outputLogsPath)
        val (newValMicroF1, newValMacroF1) = measure(
          validDataset,
          extended = evaluationLogExtended,
          enableOutputLogs = enableOutputLogs,
          outputLogsPath = outputLogsPath,
          batchSize = batchSize,
          uuid = uuid)
        if (useBestModel && bestModelMetric == ModelMetrics.valMicroF1) {
          if (newValMicroF1 >= lastValMicroF1) {
            lastCheckPoints = saveBestModel()
            lastValMicroF1 = newValMicroF1
          }
        }
      }

      if (test.nonEmpty) {
        println("Quality on test dataset: ")
        outputLog("Quality on test dataset: ", uuid, enableOutputLogs, outputLogsPath)
        val (newTestMicroF1, newTestMacroF1) = measure(
          test,
          extended = evaluationLogExtended,
          enableOutputLogs = enableOutputLogs,
          outputLogsPath = outputLogsPath,
          batchSize = batchSize,
          uuid = uuid)
        if (useBestModel && bestModelMetric == ModelMetrics.testMicroF1) {
          if (newTestMicroF1 >= lastTestMircoF1) {
            lastCheckPoints = saveBestModel()
            lastTestMircoF1 = newTestMicroF1
          }
        }
      }

      if (useBestModel && bestModelMetric == ModelMetrics.loss) {
        if (loss < lastLoss) {
          lastLoss = loss
          lastCheckPoints = saveBestModel()
        }
      }

      if (!useBestModel) {
        lastCheckPoints = tensorflow.getTFSession()
      }

    }

    if (enableOutputLogs) {
      OutputHelper.exportLogFile(outputLogsPath)
    }

    lastCheckPoints

  }

  def saveBestModel(): Session = {
    val newWrapper = new TensorflowWrapper(
      TensorflowWrapper.extractVariablesSavedModel(tensorflow.getTFSession()),
      tensorflow.graph)
    newWrapper.getTFSession()
  }

  def calcStat(tp: Int, fp: Int, fn: Int): (Float, Float, Float) = {
    val prec = tp.toFloat / (tp.toFloat + fp.toFloat)
    val rec = tp.toFloat / (tp.toFloat + fn.toFloat)
    val f1 = 2 * ((prec * rec) / (prec + rec))

    (if (prec.isNaN) 0f else prec, if (rec.isNaN) 0f else rec, if (f1.isNaN) 0 else f1)
  }

  def tagsForTokens(
      labels: Array[(String, Option[Array[Map[String, String]]])],
      pieces: WordpieceEmbeddingsSentence)
      : Array[(String, Option[Array[Map[String, String]]])] = {
    labels.zip(pieces.tokens).flatMap { case (l, p) =>
      if (p.isWordStart)
        Some(l)
      else
        None
    }
  }

  def tagsForTokens(
      labels: Array[Array[(String, Option[Array[Map[String, String]]])]],
      pieces: Array[WordpieceEmbeddingsSentence])
      : Array[Array[(String, Option[Array[Map[String, String]]])]] = {

    labels
      .zip(pieces)
      .map { case (l, p) => tagsForTokens(l, p) }
  }

  def measure(
      labeled: Iterator[Array[(TextSentenceLabels, WordpieceEmbeddingsSentence)]],
      extended: Boolean = false,
      enableOutputLogs: Boolean = false,
      outputLogsPath: String,
      batchSize: Int = 8,
      uuid: String = Identifiable.randomUID("annotator")): (Float, Float) = {

    val started = System.nanoTime()

    val predicted = mutable.Map[String, Int]()
    val correct = mutable.Map[String, Int]()

    val truePositives = mutable.Map[String, Int]()
    val falsePositives = mutable.Map[String, Int]()
    val falseNegatives = mutable.Map[String, Int]()

    for (batch <- labeled) {

      val sentencePredictedTags = predict(
        batch.map(_._2),
        configProtoBytes = None,
        includeConfidence = false,
        includeAllConfidenceScores = false,
        batchSize = batchSize)

      val sentenceTokenTags = tagsForTokens(sentencePredictedTags, batch.map(_._2))

      val sentenceTokens = batch
        .map(pair =>
          pair._2.tokens
            .filter(t => t.isWordStart)
            .map(t => t.token))
        .toList

      val sentenceLabels = batch.map(pair => pair._1.labels.toArray).toList

      (sentenceTokens, sentenceLabels, sentenceTokenTags).zipped.foreach {
        case (_, labels, tags) =>
          for (i <- labels.indices) {

            val label = labels(i)
            val tag = tags(i)

            correct(label) = correct.getOrElse(label, 0) + 1
            predicted(tag._1) = predicted.getOrElse(tag._1, 0) + 1

            // We don't really care about true negatives at the moment
            if (label == tag._1) {
              truePositives(label) = truePositives.getOrElse(label, 0) + 1
            } else if (label == "O" && tag._1 != "O") {
              falsePositives(tag._1) = falsePositives.getOrElse(tag._1, 0) + 1
            } else {
              falsePositives(tag._1) = falsePositives.getOrElse(tag._1, 0) + 1
              falseNegatives(label) = falseNegatives.getOrElse(label, 0) + 1
            }

          }
      }
    }

    val endTime = (System.nanoTime() - started) / 1e9
    println(f"time to finish evaluation: $endTime%.2fs")
    outputLog(f"time to finish evaluation: $endTime%.2fs", uuid, enableOutputLogs, outputLogsPath)

    val labels = (correct.keys ++ predicted.keys).filter(label => label != "O").toSeq.distinct
    val notEmptyLabels = labels.filter(label => label != "O" && label.nonEmpty)

    val totalTruePositives =
      truePositives.filterKeys(label => notEmptyLabels.contains(label)).values.sum
    val totalFalsePositives =
      falsePositives.filterKeys(label => notEmptyLabels.contains(label)).values.sum
    val totalFalseNegatives =
      falseNegatives.filterKeys(label => notEmptyLabels.contains(label)).values.sum

    val (prec, rec, f1) = calcStat(totalTruePositives, totalFalsePositives, totalFalseNegatives)

    if (extended) {
      println("label\t tp\t fp\t fn\t prec\t rec\t f1")
      outputLog("label\t tp\t fp\t fn\t prec\t rec\t f1", uuid, enableOutputLogs, outputLogsPath)
    }

    var totalPercByClass, totalRecByClass = 0f
    for (label <- labels) {
      val tp = truePositives.getOrElse(label, 0)
      val fp = falsePositives.getOrElse(label, 0)
      val fn = falseNegatives.getOrElse(label, 0)
      val (prec, rec, f1) = calcStat(tp, fp, fn)
      if (extended) {
        println(s"$label\t $tp\t $fp\t $fn\t $prec\t $rec\t $f1")
        outputLog(
          s"$label\t $tp\t $fp\t $fn\t $prec\t $rec\t $f1",
          uuid,
          enableOutputLogs,
          outputLogsPath)
      }
      totalPercByClass = totalPercByClass + prec
      totalRecByClass = totalRecByClass + rec
    }
    val macroPercision = totalPercByClass / notEmptyLabels.length
    val macroRecall = totalRecByClass / notEmptyLabels.length
    val macroF1 = 2 * ((macroPercision * macroRecall) / (macroPercision + macroRecall))

    if (extended) {
      println(
        s"tp: $totalTruePositives fp: $totalFalsePositives fn: $totalFalseNegatives labels: ${notEmptyLabels.length}")
      outputLog(
        s"tp: $totalTruePositives fp: $totalFalsePositives fn: $totalFalseNegatives labels: ${notEmptyLabels.length}",
        uuid,
        enableOutputLogs,
        outputLogsPath)
    }
    // ex: Precision = P1+P2/2
    println(s"Macro-average\t prec: $macroPercision, rec: $macroRecall, f1: $macroF1")
    outputLog(
      s"Macro-average\t prec: $macroPercision, rec: $macroRecall, f1: $macroF1",
      uuid,
      enableOutputLogs,
      outputLogsPath)
    // ex: Precision =  TP1+TP2/TP1+TP2+FP1+FP2
    println(s"Micro-average\t prec: $prec, rec: $rec, f1: $f1")
    outputLog(
      s"Micro-average\t prec: $prec, rec: $rec, f1: $f1",
      uuid,
      enableOutputLogs,
      outputLogsPath)

    // (Micro-average, Macro-average)
    (f1, macroF1)
  }
}
