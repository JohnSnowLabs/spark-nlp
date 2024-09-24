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

package com.johnsnowlabs.ml.ai

import com.johnsnowlabs.ml.onnx.OnnxWrapper
import com.johnsnowlabs.ml.openvino.OpenvinoWrapper
import com.johnsnowlabs.ml.tensorflow.sign.ModelSignatureConstants
import com.johnsnowlabs.ml.tensorflow.{TensorResources, TensorflowWrapper}
import com.johnsnowlabs.nlp.annotators.common.TableData
import com.johnsnowlabs.nlp.annotators.tapas.{TapasEncoder, TapasInputData}
import com.johnsnowlabs.nlp.annotators.tokenizer.wordpiece.WordpieceEncoder
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import org.tensorflow.ndarray.buffer.IntDataBuffer

import scala.collection.JavaConverters._

private[johnsnowlabs] class Tapas(
    override val tensorflowWrapper: Option[TensorflowWrapper],
    override val onnxWrapper: Option[OnnxWrapper],
    override val openvinoWrapper: Option[OpenvinoWrapper],
    override val sentenceStartTokenId: Int,
    override val sentenceEndTokenId: Int,
    configProtoBytes: Option[Array[Byte]] = None,
    tags: Map[String, Int],
    signatures: Option[Map[String, String]] = None,
    vocabulary: Map[String, Int])
    extends BertClassification(
      tensorflowWrapper = tensorflowWrapper,
      onnxWrapper = onnxWrapper,
      openvinoWrapper = openvinoWrapper,
      sentenceStartTokenId = sentenceStartTokenId,
      sentenceEndTokenId = sentenceEndTokenId,
      configProtoBytes = configProtoBytes,
      tags = tags,
      signatures = signatures,
      vocabulary = vocabulary) {

  def tagTapasSpan(batch: Seq[TapasInputData]): (Array[Array[Float]], Array[Int]) = {

    val tensors = new TensorResources()

    val maxSentenceLength = batch.head.inputIds.length
    val batchLength = batch.length

    val tokenBuffers: IntDataBuffer = tensors.createIntBuffer(batchLength * maxSentenceLength)
    val maskBuffers: IntDataBuffer = tensors.createIntBuffer(batchLength * maxSentenceLength)
    val segmentBuffers: IntDataBuffer =
      tensors.createIntBuffer(batchLength * maxSentenceLength * 7)

    batch.zipWithIndex
      .foreach { case (input, idx) =>
        val offset = idx * maxSentenceLength
        tokenBuffers.offset(offset).write(input.inputIds)
        maskBuffers.offset(offset).write(input.attentionMask)
        val segmentOffset = idx * maxSentenceLength * 7

        (0 until maxSentenceLength).foreach(pos => {
          val tokenTypeOffset = segmentOffset + pos * 7
          segmentBuffers
            .offset(tokenTypeOffset)
            .write(Array(
              input.segmentIds(pos),
              input.columnIds(pos),
              input.rowIds(pos),
              input.prevLabels(pos),
              input.columnRanks(pos),
              input.invertedColumnRanks(pos),
              input.numericRelations(pos)))
        })
      }

    val session = tensorflowWrapper.get.getTFSessionWithSignature(
      configProtoBytes = configProtoBytes,
      savedSignatures = signatures,
      initAllTables = false)
    val runner = session.runner

    val tokenTensors =
      tensors.createIntBufferTensor(Array(batchLength.toLong, maxSentenceLength), tokenBuffers)
    val maskTensors =
      tensors.createIntBufferTensor(Array(batchLength.toLong, maxSentenceLength), maskBuffers)
    val segmentTensors = tensors.createIntBufferTensor(
      Array(batchLength.toLong, maxSentenceLength, 7),
      segmentBuffers)

    runner
      .feed(
        _tfBertSignatures.getOrElse(ModelSignatureConstants.InputIds.key, "missing_input_id_key"),
        tokenTensors)
      .feed(
        _tfBertSignatures.getOrElse(
          ModelSignatureConstants.AttentionMask.key,
          "missing_input_mask_key"),
        maskTensors)
      .feed(
        _tfBertSignatures.getOrElse(
          ModelSignatureConstants.TokenTypeIds.key,
          "missing_segment_ids_key"),
        segmentTensors)
      .fetch(_tfBertSignatures
        .getOrElse(ModelSignatureConstants.TapasLogitsOutput.key, "missing_end_logits_key"))
      .fetch(
        _tfBertSignatures
          .getOrElse(
            ModelSignatureConstants.TapasLogitsAggregationOutput.key,
            "missing_start_logits_key"))

    val outs = runner.run().asScala
    val logitsRaw = TensorResources.extractFloats(outs.head)
    val aggregationRaw = TensorResources.extractFloats(outs.last)

    outs.foreach(_.close())
    tensors.clearSession(outs)
    tensors.clearTensors()

    val probabilitiesDim = logitsRaw.length / batchLength
    val flatMask = batch.flatMap(_.attentionMask)
    val probabilities: Array[Array[Float]] = logitsRaw
      .map(x => if (x < -88.7f) -88.7f else x)
      .zipWithIndex
      .map { case (logit, logitIdx) =>
        (1 / (1 + math.exp(-logit).toFloat)) * flatMask(logitIdx)
      }
      .grouped(probabilitiesDim)
      .toArray

    val aggregationDim = aggregationRaw.length / batchLength
    val aggregations: Array[Int] = aggregationRaw
      .grouped(aggregationDim)
      .map(batchLogitAggregations => {
        batchLogitAggregations.zipWithIndex.maxBy(_._1)._2
      })
      .toArray

    (probabilities, aggregations)
  }

  def predictTapasSpan(
      questions: Seq[Annotation],
      tableAnnotation: Annotation,
      maxSentenceLength: Int,
      caseSensitive: Boolean,
      minCellProbability: Float): Seq[Annotation] = {

    val encoder = new WordpieceEncoder(vocabulary)
    val tapasEncoder = new TapasEncoder(sentenceStartTokenId, sentenceEndTokenId, encoder)

    val table = TableData.fromJson(tableAnnotation.result)

    val tapasData = tapasEncoder.encodeTapasData(
      questions = questions.map(_.result),
      table = table,
      caseSensitive = caseSensitive,
      maxSentenceLength = maxSentenceLength)

    val (probabilities, aggregations) = tagTapasSpan(batch = tapasData)

    val cellPredictions = tapasData.zipWithIndex
      .flatMap { case (input, idx) =>
        val maxWidth = input.columnIds.max
        val maxHeight = input.rowIds.max
        if (maxWidth > 0 || maxHeight > 0) {
          val coordToProbabilities = collection.mutable.Map[(Int, Int), Array[Float]]()
          probabilities(idx).zipWithIndex.foreach { case (prob, probIdx) =>
            if (input.segmentIds(probIdx) == 1 && input.columnIds(probIdx) > 0 && input.rowIds(
                probIdx) > 0) {
              val coord = (input.columnIds(probIdx) - 1, input.rowIds(probIdx) - 1)
              coordToProbabilities(coord) =
                coordToProbabilities.getOrElse(coord, Array()) ++ Array(prob)
            }
          }
          val meanCoordProbs = coordToProbabilities.map(x => (x._1, x._2.sum / x._2.length))
          val answerCoordinates = collection.mutable.ArrayBuffer[(Int, Int)]()
          val answerScores = collection.mutable.ArrayBuffer[Float]()
          input.columnIds.indices.foreach { col =>
            input.rowIds.indices.foreach { row =>
              val score = meanCoordProbs.getOrElse((col, row), -1f)
              if (meanCoordProbs.getOrElse((col, row), -1f) > minCellProbability) {
                answerCoordinates.append((col, row))
                answerScores.append(score)
              }
            }
          }
          val results = answerCoordinates.zip(answerScores).sortBy(_._1).toArray
          Seq(results)
        } else {
          Seq()
        }
      }
    cellPredictions.zipWithIndex
      .map { case (result, queryId) =>
        val cellPositions = result.map(_._1)
        val scores = result.map(_._2)
        val answers = cellPositions.map(x => table.rows(x._2)(x._1)).mkString(", ")
        val aggrString = tapasEncoder.getAggregationString(aggregations(queryId))
        val resultString = if (aggrString != "NONE") s"$aggrString($answers)" else answers
        new Annotation(
          annotatorType = AnnotatorType.CHUNK,
          begin = 0,
          end = resultString.length,
          result = resultString,
          metadata = Map(
            "question" -> questions(queryId).result,
            "aggregation" -> aggrString,
            "cell_positions" ->
              cellPositions.map(x => "[%d, %d]".format(x._1, x._2)).mkString(", "),
            "cell_scores" -> scores.map(_.toString).mkString(", ")))
      }
  }
}
