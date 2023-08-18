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

package com.johnsnowlabs.ml.ai.util

import com.johnsnowlabs.ml.tensorflow.TensorResources
import com.johnsnowlabs.nlp.annotators.common._
import org.tensorflow.Tensor
import org.tensorflow.ndarray.buffer.IntDataBuffer

private[johnsnowlabs] object PrepareEmbeddings {

  /** prepare batches of piece IDs while padding shorter sequences to the longest sequence length
    * and trim sequences longer than maxSequenceLength
    *
    * @param sentences
    *   batches of WordpieceTokenizedSentence
    * @param maxSequenceLength
    *   provided maximum allowed sequence length
    * @param sentenceStartTokenId
    *   id for token piece at the beginning of sequence
    * @param sentenceEndTokenId
    *   id for token piece at the end of sequence
    * @param sentencePadTokenId
    *   id for token piece used for padding
    * @return
    */
  def prepareBatchInputsWithPadding(
      sentences: Seq[(WordpieceTokenizedSentence, Int)],
      maxSequenceLength: Int,
      sentenceStartTokenId: Int,
      sentenceEndTokenId: Int,
      sentencePadTokenId: Int = 0): Seq[Array[Int]] = {
    val maxSentenceLength =
      Array(
        maxSequenceLength - 2,
        sentences.map { case (wpTokSentence, _) =>
          wpTokSentence.tokens.length
        }.max).min

    sentences
      .map { case (wpTokSentence, _) =>
        val tokenPieceIds = wpTokSentence.tokens.map(t => t.pieceId)
        val padding = Array.fill(maxSentenceLength - tokenPieceIds.length)(sentencePadTokenId)

        Array(sentenceStartTokenId) ++ tokenPieceIds.take(maxSentenceLength) ++ Array(
          sentenceEndTokenId) ++ padding
      }
  }

  def prepareOvLongBatchTensors(
      batch: Seq[Array[Int]],
      maxSentenceLength: Int,
      batchLength: Int,
      sentencePadTokenId: Int = 0): (org.intel.openvino.Tensor, org.intel.openvino.Tensor) = {
    val shape = Array(batchLength, maxSentenceLength)
    val tokenTensors =
      new org.intel.openvino.Tensor(shape, batch.flatMap(x => x.map(x => x.toLong)).toArray)
    val maskTensors = new org.intel.openvino.Tensor(
      shape,
      batch
        .flatMap(sentence => sentence.map(x => if (x == sentencePadTokenId) 0L else 1L))
        .toArray)

    (tokenTensors, maskTensors)
  }

  def prepareOvIntBatchTensorsWithSegment(
      batch: Seq[Array[Int]],
      maxSentenceLength: Int,
      batchLength: Int,
      sentencePadTokenId: Int = 0)
      : (org.intel.openvino.Tensor, org.intel.openvino.Tensor, org.intel.openvino.Tensor) = {
    val shape = Array(batchLength, maxSentenceLength)
    val tokenTensors =
      new org.intel.openvino.Tensor(shape, batch.flatten.toArray)
    val maskTensors = new org.intel.openvino.Tensor(
      shape,
      batch
        .flatMap(sentence => sentence.map(x => if (x == sentencePadTokenId) 0L else 1L))
        .toArray)
    val segmentTensors =
      new org.intel.openvino.Tensor(shape, Array.fill(batchLength * maxSentenceLength)(0))

    (tokenTensors, maskTensors, segmentTensors)
  }

  def prepareBatchTensors(
      tensors: TensorResources,
      batch: Seq[Array[Int]],
      maxSentenceLength: Int,
      batchLength: Int,
      sentencePadTokenId: Int = 0): (Tensor, Tensor) = {

    val tokenBuffers: IntDataBuffer = tensors.createIntBuffer(batchLength * maxSentenceLength)
    val maskBuffers: IntDataBuffer = tensors.createIntBuffer(batchLength * maxSentenceLength)

    batch.zipWithIndex
      .foreach { case (sentence, idx) =>
        val offset = idx * maxSentenceLength
        tokenBuffers.offset(offset).write(sentence)
        maskBuffers.offset(offset).write(sentence.map(x => if (x == sentencePadTokenId) 0 else 1))
      }

    // [nb of encoded sentences , maxSentenceLength]
    val shape = Array(batch.length.toLong, maxSentenceLength)

    val tokenTensors = tensors.createIntBufferTensor(shape, tokenBuffers)
    val maskTensors = tensors.createIntBufferTensor(shape, maskBuffers)

    (tokenTensors, maskTensors)

  }

  def prepareBatchTensorsWithSegment(
      tensors: TensorResources,
      batch: Seq[Array[Int]],
      maxSentenceLength: Int,
      batchLength: Int,
      sentencePadTokenId: Int = 0): (Tensor, Tensor, Tensor) = {

    val tokenBuffers: IntDataBuffer = tensors.createIntBuffer(batchLength * maxSentenceLength)
    val maskBuffers: IntDataBuffer = tensors.createIntBuffer(batchLength * maxSentenceLength)
    val segmentBuffers: IntDataBuffer = tensors.createIntBuffer(batchLength * maxSentenceLength)

    batch.zipWithIndex
      .foreach { case (sentence, idx) =>
        val offset = idx * maxSentenceLength
        tokenBuffers.offset(offset).write(sentence)
        maskBuffers.offset(offset).write(sentence.map(x => if (x == sentencePadTokenId) 0 else 1))
        segmentBuffers.offset(offset).write(Array.fill(maxSentenceLength)(0))
      }

    // [nb of encoded sentences , maxSentenceLength]
    val shape = Array(batch.length.toLong, maxSentenceLength)

    val tokenTensors = tensors.createIntBufferTensor(shape, tokenBuffers)
    val maskTensors = tensors.createIntBufferTensor(shape, maskBuffers)
    val segmentTensors = tensors.createIntBufferTensor(shape, segmentBuffers)

    (tokenTensors, maskTensors, segmentTensors)

  }

  def prepareBatchWordEmbeddings(
      batch: Seq[Array[Int]],
      embeddings: Array[Float],
      maxSentenceLength: Int,
      batchLength: Int): Seq[Array[Array[Float]]] = {
    val dim = embeddings.length / (batchLength * maxSentenceLength)
    val batchEmbeddings: Array[Array[Array[Float]]] =
      embeddings.grouped(dim).toArray.grouped(maxSentenceLength).toArray

    val emptyVector = Array.fill(dim)(0f)

    batch.zip(batchEmbeddings).map { case (ids, embeddings) =>
      if (ids.length > embeddings.length) {
        embeddings.take(embeddings.length - 1) ++
          Array.fill(embeddings.length - ids.length)(emptyVector) ++
          Array(embeddings.last)
      } else {
        embeddings
      }
    }
  }

}
