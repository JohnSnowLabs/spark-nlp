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

import ai.onnxruntime.OnnxTensor
import com.johnsnowlabs.ml.ai.util.PrepareEmbeddings
import com.johnsnowlabs.ml.onnx.{OnnxSession, OnnxWrapper}
import com.johnsnowlabs.ml.openvino.OpenvinoWrapper
import com.johnsnowlabs.ml.tensorflow.sentencepiece._
import com.johnsnowlabs.ml.tensorflow.sign.{ModelSignatureConstants, ModelSignatureManager}
import com.johnsnowlabs.ml.tensorflow.{TensorResources, TensorflowWrapper}
import com.johnsnowlabs.ml.util.{ONNX, Openvino, TensorFlow}
import com.johnsnowlabs.nlp.annotators.common._
import org.intel.openvino.Tensor
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.JavaConverters._

/** @param tensorflowWrapper
  *   DeBERTa Model wrapper with TensorFlowWrapper
  * @param spp
  *   DeBERTa SentencePiece model with SentencePieceWrapper
  * @param batchSize
  *   size of batch
  * @param configProtoBytes
  *   Configuration for TensorFlow session
  */
class DeBerta(
    val tensorflowWrapper: Option[TensorflowWrapper],
    val onnxWrapper: Option[OnnxWrapper],
    val openvinoWrapper: Option[OpenvinoWrapper],
    val spp: SentencePieceWrapper,
    batchSize: Int,
    configProtoBytes: Option[Array[Byte]] = None,
    signatures: Option[Map[String, String]] = None)
    extends Serializable {

  protected val logger: Logger = LoggerFactory.getLogger("Deberta")
  val _tfDeBertaSignatures: Map[String, String] =
    signatures.getOrElse(ModelSignatureManager.apply())

  val detectedEngine: String =
    if (tensorflowWrapper.isDefined) TensorFlow.name
    else if (onnxWrapper.isDefined) ONNX.name
    else if (openvinoWrapper.isDefined) Openvino.name
    else TensorFlow.name
  private val onnxSessionOptions: Map[String, String] = new OnnxSession().getSessionOptions

  // keys representing the input and output tensors of the DeBERTa model
  private val SentenceStartTokenId = spp.getSppModel.pieceToId("[CLS]")
  private val SentenceEndTokenId = spp.getSppModel.pieceToId("[SEP]")
  private val SentencePadTokenId = spp.getSppModel.pieceToId("[PAD]")
  private val SentencePieceDelimiterId = spp.getSppModel.pieceToId("▁")

  def tag(batch: Seq[Array[Int]]): Seq[Array[Array[Float]]] = {
    /* Actual size of each sentence to skip padding in the TF model */
    val maxSentenceLength = batch.map(pieceIds => pieceIds.length).max
    val batchLength = batch.length

    val embeddings = detectedEngine match {

      case ONNX.name =>
        // [nb of encoded sentences , maxSentenceLength]
        val (runner, env) = onnxWrapper.get.getSession(onnxSessionOptions)

        val tokenTensors =
          OnnxTensor.createTensor(env, batch.map(x => x.map(x => x.toLong)).toArray)
        val maskTensors =
          OnnxTensor.createTensor(
            env,
            batch.map(sentence => sentence.map(x => if (x == 0L) 0L else 1L)).toArray)

        val segmentTensors =
          OnnxTensor.createTensor(env, batch.map(x => Array.fill(maxSentenceLength)(0L)).toArray)

        val inputs =
          Map("input_ids" -> tokenTensors, "attention_mask" -> maskTensors).asJava

        try {
          val results = runner.run(inputs)
          try {
            val embeddings = results
              .get("last_hidden_state")
              .get()
              .asInstanceOf[OnnxTensor]
              .getFloatBuffer
              .array()

            embeddings
          } finally if (results != null) results.close()
        } catch {
          case e: Exception =>
            // Handle exceptions by logging or other means.
            e.printStackTrace()
            Array.empty[Float] // Return an empty array or appropriate error handling
        } finally {
          // Close tensors outside the try-catch to avoid repeated null checks.
          // These resources are initialized before the try-catch, so they should be closed here.
          tokenTensors.close()
          maskTensors.close()
          segmentTensors.close()
        }



      case Openvino.name =>

        val batchLength = batch.length
        val shape = Array(batchLength, maxSentenceLength)
        val (tokenTensors, maskTensors) =
          PrepareEmbeddings.prepareOvLongBatchTensors(batch, maxSentenceLength, batchLength)

        val inferRequest = openvinoWrapper.get.getCompiledModel().create_infer_request()
        inferRequest.set_tensor("input_ids", tokenTensors)
        inferRequest.set_tensor("attention_mask", maskTensors)

        inferRequest.infer()

        try {
          try {
            inferRequest
              .get_tensor("last_hidden_state")
              .data()
          }
        } catch {
          case e: Exception =>
            e.printStackTrace()
            Array.empty[Float]
            // Rethrow the exception to propagate it further
            throw e
        }


      case _ =>
        val tensors = new TensorResources()

        val (tokenTensors, maskTensors, segmentTensors) =
          PrepareEmbeddings.prepareBatchTensorsWithSegment(
            tensors,
            batch,
            maxSentenceLength,
            batchLength)

        val runner = tensorflowWrapper.get
          .getTFSessionWithSignature(
            configProtoBytes = configProtoBytes,
            savedSignatures = signatures,
            initAllTables = false)
          .runner

        runner
          .feed(
            _tfDeBertaSignatures.getOrElse(
              ModelSignatureConstants.InputIds.key,
              "missing_input_id_key"),
            tokenTensors)
          .feed(
            _tfDeBertaSignatures
              .getOrElse(ModelSignatureConstants.AttentionMask.key, "missing_input_mask_key"),
            maskTensors)
          .feed(
            _tfDeBertaSignatures
              .getOrElse(ModelSignatureConstants.TokenTypeIds.key, "missing_segment_ids_key"),
            segmentTensors)
          .fetch(
            _tfDeBertaSignatures
              .getOrElse(
                ModelSignatureConstants.LastHiddenState.key,
                "missing_sequence_output_key"))

        val outs = runner.run().asScala
        val embeddings = TensorResources.extractFloats(outs.head)

        tokenTensors.close()
        maskTensors.close()
        segmentTensors.close()
        tensors.clearSession(outs)
        tensors.clearTensors()

        embeddings

    }

    PrepareEmbeddings.prepareBatchWordEmbeddings(
      batch,
      embeddings,
      maxSentenceLength,
      batchLength)
  }

  def predict(
      tokenizedSentences: Seq[TokenizedSentence],
      batchSize: Int,
      maxSentenceLength: Int,
      caseSensitive: Boolean): Seq[WordpieceEmbeddingsSentence] = {

    val wordPieceTokenizedSentences =
      tokenizeWithAlignment(tokenizedSentences, maxSentenceLength, caseSensitive)
    wordPieceTokenizedSentences.zipWithIndex
      .grouped(batchSize)
      .flatMap { batch =>
        val encoded = PrepareEmbeddings.prepareBatchInputsWithPadding(
          batch,
          maxSentenceLength,
          SentenceStartTokenId,
          SentenceEndTokenId,
          SentencePadTokenId)
        val vectors = tag(encoded)

        /*Combine tokens and calculated embeddings*/
        batch.zip(vectors).map { case (sentence, tokenVectors) =>
          val tokenLength = sentence._1.tokens.length
          /*All wordpiece embeddings*/
          val tokenEmbeddings = tokenVectors.slice(1, tokenLength + 1)
          val originalIndexedTokens = tokenizedSentences(sentence._2)

          val tokensWithEmbeddings =
            sentence._1.tokens.zip(tokenEmbeddings).flatMap { case (token, tokenEmbedding) =>
              val tokenWithEmbeddings = TokenPieceEmbeddings(token, tokenEmbedding)
              val originalTokensWithEmbeddings = originalIndexedTokens.indexedTokens
                .find(p =>
                  p.begin == tokenWithEmbeddings.begin && tokenWithEmbeddings.isWordStart)
                .map { token =>
                  val originalTokenWithEmbedding = TokenPieceEmbeddings(
                    TokenPiece(
                      wordpiece = tokenWithEmbeddings.wordpiece,
                      token = if (caseSensitive) token.token else token.token.toLowerCase(),
                      pieceId = tokenWithEmbeddings.pieceId,
                      isWordStart = tokenWithEmbeddings.isWordStart,
                      begin = token.begin,
                      end = token.end),
                    tokenEmbedding)
                  originalTokenWithEmbedding
                }
              originalTokensWithEmbeddings
            }

          WordpieceEmbeddingsSentence(tokensWithEmbeddings, originalIndexedTokens.sentenceIndex)
        }
      }
      .toSeq
  }

  def tokenizeWithAlignment(
      sentences: Seq[TokenizedSentence],
      maxSeqLength: Int,
      caseSensitive: Boolean): Seq[WordpieceTokenizedSentence] = {
    val encoder =
      new SentencepieceEncoder(spp, caseSensitive, delimiterId = SentencePieceDelimiterId)

    val sentenceTokenPieces = sentences.map { s =>
      val trimmedSentence = s.indexedTokens.take(maxSeqLength - 2)
      val wordpieceTokens =
        trimmedSentence.flatMap(token => encoder.encode(token)).take(maxSeqLength)
      WordpieceTokenizedSentence(wordpieceTokens)
    }
    sentenceTokenPieces
  }

}
