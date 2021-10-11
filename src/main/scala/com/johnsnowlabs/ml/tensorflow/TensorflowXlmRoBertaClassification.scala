/*
 * Copyright 2017-2021 John Snow Labs
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

import com.johnsnowlabs.ml.tensorflow.sentencepiece.{SentencePieceWrapper, SentencepieceEncoder}
import com.johnsnowlabs.ml.tensorflow.sign.{ModelSignatureConstants, ModelSignatureManager}
import com.johnsnowlabs.nlp.annotators.common._
import org.tensorflow.ndarray.buffer.IntDataBuffer

import scala.collection.JavaConverters._

/**
 *
 * @param tensorflowWrapper XLM-RoBERTa Model wrapper with TensorFlow Wrapper
 * @param spp               XlmRoberta SentencePiece model with SentencePieceWrapper
 * @param configProtoBytes  Configuration for TensorFlow session
 * @param tags              labels which model was trained with in order
 * @param signatures        TF v2 signatures in Spark NLP
 * */
class TensorflowXlmRoBertaClassification(val tensorflowWrapper: TensorflowWrapper,
                                         val spp: SentencePieceWrapper,
                                         configProtoBytes: Option[Array[Byte]] = None,
                                         tags: Map[String, Int],
                                         signatures: Option[Map[String, String]] = None
                                        ) extends Serializable with TensorflowForClassification {

  val _tfXlmRoBertaSignatures: Map[String, String] = signatures.getOrElse(ModelSignatureManager.apply())

  protected val sentenceStartTokenId: Int = 0
  protected val sentenceEndTokenId: Int = 2
  protected val sentencePadTokenId: Int = 1

  private val sentencePieceDelimiterId = spp.getSppModel.pieceToId("▁")

  def tokenizeWithAlignment(sentences: Seq[TokenizedSentence], maxSeqLength: Int, caseSensitive: Boolean):
  Seq[WordpieceTokenizedSentence] = {

    val encoder = new SentencepieceEncoder(spp, caseSensitive, sentencePieceDelimiterId, pieceIdFromZero = true)

    val sentecneTokenPieces = sentences.map { s =>
      val shrinkedSentence = s.indexedTokens.take(maxSeqLength - 2)
      val wordpieceTokens = shrinkedSentence.flatMap(token => encoder.encode(token)).take(maxSeqLength)
      WordpieceTokenizedSentence(wordpieceTokens)
    }
    sentecneTokenPieces
  }

  def tag(batch: Seq[Array[Int]]): Seq[Array[Array[Float]]] = {
    val tensors = new TensorResources()

    val maxSentenceLength = batch.map(encodedSentence => encodedSentence.length).max
    val batchLength = batch.length

    val tokenBuffers: IntDataBuffer = tensors.createIntBuffer(batchLength * maxSentenceLength)
    val maskBuffers: IntDataBuffer = tensors.createIntBuffer(batchLength * maxSentenceLength)

    // [nb of encoded sentences , maxSentenceLength]
    val shape = Array(batch.length.toLong, maxSentenceLength)

    batch.zipWithIndex
      .foreach { case (sentence, idx) =>
        val offset = idx * maxSentenceLength
        tokenBuffers.offset(offset).write(sentence)
        maskBuffers.offset(offset).write(sentence.map(x => if (x == sentencePadTokenId) 0 else 1))
      }

    val runner = tensorflowWrapper.getTFHubSession(configProtoBytes = configProtoBytes, initAllTables = false).runner

    val tokenTensors = tensors.createIntBufferTensor(shape, tokenBuffers)
    val maskTensors = tensors.createIntBufferTensor(shape, maskBuffers)

    runner
      .feed(_tfXlmRoBertaSignatures.getOrElse(ModelSignatureConstants.InputIds.key, "missing_input_id_key"), tokenTensors)
      .feed(_tfXlmRoBertaSignatures.getOrElse(ModelSignatureConstants.AttentionMask.key, "missing_input_mask_key"), maskTensors)
      .fetch(_tfXlmRoBertaSignatures.getOrElse(ModelSignatureConstants.LogitsOutput.key, "missing_logits_key"))

    val outs = runner.run().asScala
    val rawScores = TensorResources.extractFloats(outs.head)

    outs.foreach(_.close())
    tensors.clearSession(outs)
    tensors.clearTensors()

    val dim = rawScores.length / (batchLength * maxSentenceLength)
    val batchScores: Array[Array[Array[Float]]] = rawScores.grouped(dim).map(scores =>
      calculateSoftmax(scores)).toArray.grouped(maxSentenceLength).toArray

    batchScores
  }

  def findIndexedToken(tokenizedSentences: Seq[TokenizedSentence], sentence: (WordpieceTokenizedSentence, Int),
                                tokenPiece: TokenPiece): Option[IndexedToken] = {
    tokenizedSentences(sentence._2).indexedTokens.find(p => p.begin == tokenPiece.begin && tokenPiece.isWordStart)
  }

}


