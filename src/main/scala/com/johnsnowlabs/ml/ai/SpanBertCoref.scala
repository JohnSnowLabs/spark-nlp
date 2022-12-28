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

import com.johnsnowlabs.ml.tensorflow.sign.ModelSignatureManager
import com.johnsnowlabs.ml.tensorflow.{TensorResources, TensorflowWrapper}
import org.tensorflow.ndarray.buffer.IntDataBuffer

import scala.collection.JavaConverters._

class SpanBertCoref(
    val tensorflowWrapper: TensorflowWrapper,
    sentenceStartTokenId: Int,
    sentenceEndTokenId: Int,
    configProtoBytes: Option[Array[Byte]] = None,
    signatures: Option[Map[String, String]] = None)
    extends Serializable {

  val _tfSpanBertCorefSignatures: Map[String, String] =
    signatures.getOrElse(ModelSignatureManager.apply())

  def predict(
      inputIds: Array[Array[Int]],
      genre: Int,
      maxSegmentLength: Int): Array[Array[((Int, Int), (Int, Int))]] = {
    val tensors = new TensorResources()

    val tokenIndices = scala.collection.mutable.Map[Int, (Int, Int)]()

    val paddedInputIds =
      scala.collection.mutable.ArrayBuffer[Array[Int]](Array(sentenceStartTokenId))
    val paddedInputMasks = scala.collection.mutable.ArrayBuffer[Array[Int]](Array(1))
    val paddedTextLengths = scala.collection.mutable.ArrayBuffer[Int](1)
    val paddedSpeakerIds = scala.collection.mutable.ArrayBuffer[Array[Int]](Array(1))
    val paddedSentenceMap = scala.collection.mutable.ArrayBuffer[Int](0)

    inputIds.zipWithIndex.foreach { case (sentenceInputIds, sentenceNo) =>
      val currentIdx = paddedInputIds.length - 1
      if ((paddedInputIds(currentIdx).length + sentenceInputIds.length) > maxSegmentLength) {
        paddedInputIds(currentIdx) = paddedInputIds(currentIdx) ++ Array(sentenceEndTokenId)
        paddedInputMasks(currentIdx) = paddedInputMasks(currentIdx) ++ Array(1)
        paddedInputIds.append(Array(sentenceStartTokenId))
        paddedInputMasks.append(Array(1))
        paddedTextLengths(currentIdx) = paddedInputIds(currentIdx).length
        paddedTextLengths.append(1)
        paddedSpeakerIds(currentIdx) = paddedSpeakerIds(currentIdx) ++ Array(1)
        paddedSentenceMap.append(sentenceNo - 1)
        paddedSentenceMap.append(sentenceNo)
      }
      val currentIdx2 = paddedInputIds.length - 1
      val tokensStartIndex = paddedInputIds.map(_.length).sum
      paddedInputIds(currentIdx2) = paddedInputIds(currentIdx2) ++ sentenceInputIds
      paddedInputMasks(currentIdx2) = paddedInputIds(currentIdx2).map(_ => 1)
      paddedSpeakerIds(currentIdx2) =
        paddedSpeakerIds(currentIdx2) ++ sentenceInputIds.map(_ => 2)
      sentenceInputIds.foreach(_ => paddedSentenceMap.append(sentenceNo))
      paddedTextLengths(currentIdx2) = paddedInputIds(currentIdx2).length

      sentenceInputIds.indices.foreach { i =>
        tokenIndices(tokensStartIndex + i) = (sentenceNo, i)
      }

    }

    // finalize last sentence
    val lastIdx = paddedInputIds.length - 1
    paddedInputIds(lastIdx) = paddedInputIds(lastIdx) ++ Array(sentenceEndTokenId)
    paddedInputMasks(lastIdx) = paddedInputMasks(lastIdx) ++ Array(1)
    paddedTextLengths(lastIdx) = paddedInputIds(lastIdx).length
    paddedSpeakerIds(lastIdx) = paddedSpeakerIds(lastIdx) ++ Array(1)
    paddedSentenceMap.append(inputIds.length - 1)

    // pad
    paddedInputIds.indices.foreach(currentIdx => {
      paddedInputIds(currentIdx) = paddedInputIds(currentIdx).padTo(maxSegmentLength, 0)
      paddedInputMasks(currentIdx) = paddedInputMasks(currentIdx).padTo(maxSegmentLength, 0)
      paddedSpeakerIds(currentIdx) = paddedSpeakerIds(currentIdx).padTo(maxSegmentLength, 0)
    })

    val batchSize = paddedInputIds.length

    val inputIdsBuffer: IntDataBuffer = tensors.createIntBuffer(batchSize * maxSegmentLength)
    val inputMaskBuffer: IntDataBuffer = tensors.createIntBuffer(batchSize * maxSegmentLength)
    val textLengthBuffer: IntDataBuffer = tensors.createIntBuffer(batchSize)
    val speakerIdsBuffer: IntDataBuffer = tensors.createIntBuffer(batchSize * maxSegmentLength)
    val goldStartsBuffer: IntDataBuffer = tensors.createIntBuffer(0)
    val goldEndsBuffer: IntDataBuffer = tensors.createIntBuffer(0)
    val clusterIdsBuffer: IntDataBuffer = tensors.createIntBuffer(0)
    val sentenceMapBuffer: IntDataBuffer = tensors.createIntBuffer(paddedSentenceMap.length)

    inputIdsBuffer.write(paddedInputIds.toArray.flatten)
    inputMaskBuffer.write(paddedInputMasks.toArray.flatten)
    textLengthBuffer.write(paddedTextLengths.toArray)
    speakerIdsBuffer.write(paddedSpeakerIds.toArray.flatten)
    sentenceMapBuffer.write(paddedSentenceMap.toArray)

    val runner = tensorflowWrapper
      .getTFSessionWithSignature(
        configProtoBytes = configProtoBytes,
        savedSignatures = signatures,
        initAllTables = false)
      .runner

    val inputIdsShape = Array(batchSize.toLong, maxSegmentLength.toLong)
    val singleValueShape = Array(batchSize.toLong)
    val emptyShape = Array(0L)
    val sentenceMapShape = Array(paddedSentenceMap.length.toLong)

    val inputIdsTensors = tensors.createIntBufferTensor(inputIdsShape, inputIdsBuffer)
    val inputMaskTensors = tensors.createIntBufferTensor(inputIdsShape, inputMaskBuffer)
    val textLengthTensors = tensors.createIntBufferTensor(singleValueShape, textLengthBuffer)
    val speakerIdsTensors = tensors.createIntBufferTensor(inputIdsShape, speakerIdsBuffer)
    val genreTensors = tensors.createTensor(genre)
    val isTrainingTensors = tensors.createTensor(false)
    val goldStartsTensors = tensors.createIntBufferTensor(emptyShape, goldStartsBuffer)
    val goldEndTensors = tensors.createIntBufferTensor(emptyShape, goldEndsBuffer)
    val clusterTensors = tensors.createIntBufferTensor(emptyShape, clusterIdsBuffer)
    val sentenceMapTensors = tensors.createIntBufferTensor(sentenceMapShape, sentenceMapBuffer)

    runner
      .feed(
        _tfSpanBertCorefSignatures.getOrElse("input_ids", "missing_input_id_key"),
        inputIdsTensors)
      .feed(
        _tfSpanBertCorefSignatures.getOrElse("attention_mask", "missing_attention_mask_key"),
        inputMaskTensors)
      .feed(
        _tfSpanBertCorefSignatures.getOrElse("text_lens", "missing_text_lens_key"),
        textLengthTensors)
      .feed(
        _tfSpanBertCorefSignatures.getOrElse("speaker_ids", "missing_speaker_ids_key"),
        speakerIdsTensors)
      .feed(_tfSpanBertCorefSignatures.getOrElse("genre", "missing_genre_key"), genreTensors)
      .feed(
        _tfSpanBertCorefSignatures.getOrElse("is_training", "missing_is_training_key"),
        isTrainingTensors)
      .feed(
        _tfSpanBertCorefSignatures.getOrElse("gold_starts", "missing_gold_starts_key"),
        goldStartsTensors)
      .feed(
        _tfSpanBertCorefSignatures.getOrElse("gold_ends", "missing_gold_ends_key"),
        goldEndTensors)
      .feed(
        _tfSpanBertCorefSignatures.getOrElse("cluster_ids", "missing_cluster_ids_key"),
        clusterTensors)
      .feed(
        _tfSpanBertCorefSignatures.getOrElse("sentence_map", "missing_sentence_map_key"),
        sentenceMapTensors)
      .fetch(_tfSpanBertCorefSignatures
        .getOrElse("candidate_mention_scores_t", "missing_candidate_mention_scores_t_key"))
      .fetch(_tfSpanBertCorefSignatures
        .getOrElse("candidate_starts_t", "missing_candidate_starts_t_key"))
      .fetch(_tfSpanBertCorefSignatures
        .getOrElse("candidate_ends_t", "missing_candidate_ends_t_key"))
      .fetch(_tfSpanBertCorefSignatures.getOrElse("k_t", "missing_k_t_key"))
      .fetch(_tfSpanBertCorefSignatures.getOrElse("num_words_t", "missing_num_words_t_key"))

    val t_results = runner.run()

    val spanScoresRaw = TensorResources.extractFloats(t_results.get(0))
    val numCandidateSpans = spanScoresRaw.length / batchSize
    val spanScores = spanScoresRaw.grouped(numCandidateSpans).toArray
    val candidateStarts =
      TensorResources.extractInts(t_results.get(1)).grouped(numCandidateSpans).toArray
    val candidateEnds =
      TensorResources.extractInts(t_results.get(2)).grouped(numCandidateSpans).toArray
    val numOutputSpans = TensorResources.extractInts(t_results.get(3))
    val numWords = TensorResources.extractInt(t_results.get(4))

    val topSpanIndices =
      extractSpans(spanScores, candidateStarts, candidateEnds, numOutputSpans, numWords)
    val maxNumOutputSpans = numOutputSpans.max
    val topSpanIndicesBuffer: IntDataBuffer =
      tensors.createIntBuffer(batchSize * maxNumOutputSpans)
    topSpanIndices.zipWithIndex.foreach(x =>
      topSpanIndicesBuffer.offset(x._2 * maxNumOutputSpans).write(x._1))
    val topSpanIndicesTensors =
      tensors.createIntBufferTensor(Array(batchSize, maxNumOutputSpans), topSpanIndicesBuffer)

    runner
      .feed(
        _tfSpanBertCorefSignatures.getOrElse("top_span_indices", "missing_top_span_indices_key"),
        topSpanIndicesTensors)
      .fetch(
        _tfSpanBertCorefSignatures.getOrElse("top_span_starts", "missing_top_span_starts_key"))
      .fetch(_tfSpanBertCorefSignatures.getOrElse("top_span_ends", "missing_top_span_ends_key"))
      .fetch(_tfSpanBertCorefSignatures.getOrElse("top_antecedents", "missing_top_span_ends_key"))
      .fetch(_tfSpanBertCorefSignatures
        .getOrElse("top_antecedent_scores", "missing_top_antecedent_scores_key"))

    val results = runner.run()
    val topSpanStarts = TensorResources.extractInts(results.get(5))
    val topSpanEnds = TensorResources.extractInts(results.get(6))
    val topSpanAntecedents =
      TensorResources.extractInts(results.get(7)).grouped(topSpanEnds.length).toArray
    val topSpanAntecedentScoresRaw = TensorResources.extractFloats(results.get(8))
    val topSpanAntecedentScores = topSpanAntecedentScoresRaw
      .grouped(topSpanAntecedentScoresRaw.length / topSpanEnds.length)
      .toArray

    val predictedAntecedents =
      getPredictedAntecedents(topSpanAntecedents, topSpanAntecedentScores)
    val (predictedClusters, _) =
      getPredictedClusters(topSpanStarts, topSpanEnds, predictedAntecedents)

    tensors.clearSession(t_results.asScala)
    tensors.clearSession(results.asScala)
    tensors.clearTensors()

    predictedClusters
      .map(cluster =>
        cluster
          .map(xy => (tokenIndices(xy._1), tokenIndices(xy._2))))

  }
  def getPredictedAntecedents(
      antecedents: Array[Array[Int]],
      antecedentScores: Array[Array[Float]]): Array[Int] = {
    antecedentScores.zipWithIndex.map { case (spanAntecedents, i) =>
      val predictedIndex = spanAntecedents.zipWithIndex.maxBy(_._1)._2 - 1
      if (predictedIndex < 0) -1 else antecedents(i)(predictedIndex)
    }
  }

  def getPredictedClusters(
      topSpanStarts: Array[Int],
      topSpanEnds: Array[Int],
      predictedAntecedents: Array[Int]): (Array[Array[(Int, Int)]], Map[(Int, Int), Int]) = {
    val mentionToPredicted = scala.collection.mutable.Map[(Int, Int), Int]()
    val predictedClusters = scala.collection.mutable.ArrayBuffer[Array[(Int, Int)]]()

    predictedAntecedents.zipWithIndex
      .filter(_._1 >= 0)
      .foreach { case (predictedIndex, i) =>
        val predictedAntecedent = (topSpanStarts(predictedIndex), topSpanEnds(predictedIndex))
        val predictedCluster = if (mentionToPredicted.contains(predictedAntecedent)) {
          mentionToPredicted(predictedAntecedent)
        } else {
          val newPredictedCluster = predictedClusters.length
          predictedClusters.append(Array(predictedAntecedent))
          mentionToPredicted(predictedAntecedent) = newPredictedCluster
          newPredictedCluster
        }
        val mention = (topSpanStarts(i), topSpanEnds(i))
        predictedClusters(predictedCluster) =
          predictedClusters(predictedCluster) ++ Array(mention)
        mentionToPredicted(mention) = predictedCluster
      }

    (predictedClusters.toArray, mentionToPredicted.toMap)
  }

  /*
    This is a terrible C-ish implementation. When calm down and relax, rewrite in proper functional Scala.
   */
  def extractSpans(
      spanScores: Array[Array[Float]],
      candidateStarts: Array[Array[Int]],
      candidateEnds: Array[Array[Int]],
      numOutputSpans: Array[Int],
      numWords: Int): Array[Array[Int]] = {
    val maxNumOutputSpans = numOutputSpans.max
    spanScores.zipWithIndex.map { case (sentenceSpanScores, sentence_i) =>
      val candidateIdxSorted = sentenceSpanScores.zipWithIndex.sortBy(-_._1).map(_._2)
      val selectedCandidateIdx = scala.collection.mutable.ArrayBuffer[Int]()
      val startToMaxEnd = scala.collection.mutable.Map[Int, Int]()
      val endToMinStart = scala.collection.mutable.Map[Int, Int]()
      candidateIdxSorted.foreach(candidateIdx => {
        if (selectedCandidateIdx.length < maxNumOutputSpans) {
          val spanStartIdx = candidateStarts(sentence_i)(candidateIdx)
          val spanEndIdx = candidateEnds(sentence_i)(candidateIdx)
          var crossOverlap = false
          (spanStartIdx to spanEndIdx).inclusive.foreach(tokenIdx => {
            if (!crossOverlap) {
              val maxEnd = startToMaxEnd.getOrElse(tokenIdx, -1)
              if ((tokenIdx > spanStartIdx) && (maxEnd > spanEndIdx)) {
                crossOverlap = true
              }
              val minStart = endToMinStart.getOrElse(tokenIdx, -1)
              if ((tokenIdx < spanEndIdx) && (0 <= minStart) && (minStart < spanStartIdx)) {
                crossOverlap = true
              }
            }
          })
          if (!crossOverlap) {
            selectedCandidateIdx.append(candidateIdx)
            val maxEnd = startToMaxEnd.getOrElse(spanStartIdx, -1)
            if (spanEndIdx > maxEnd) {
              startToMaxEnd(spanStartIdx) = spanEndIdx
            }
            val minStart = endToMinStart.getOrElse(spanEndIdx, -1)
            if ((minStart == -1) || spanStartIdx < minStart) {
              endToMinStart(spanEndIdx) = spanStartIdx
            }
          }

        }
      })
      val sortedSelectedCandidateIdx = selectedCandidateIdx.toArray.sortWith { case (x, y) =>
        if (candidateStarts(sentence_i)(x) < candidateStarts(sentence_i)(y))
          true
        else if (candidateStarts(sentence_i)(x) > candidateStarts(sentence_i)(y))
          false
        else
          candidateEnds(sentence_i)(x) <= candidateEnds(sentence_i)(y)
      }
      sortedSelectedCandidateIdx.padTo(
        maxNumOutputSpans,
        sortedSelectedCandidateIdx.headOption.getOrElse(0))
    }

  }
}
