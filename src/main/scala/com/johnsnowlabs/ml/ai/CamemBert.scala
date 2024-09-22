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
import com.johnsnowlabs.ml.tensorflow.sentencepiece.{SentencePieceWrapper, SentencepieceEncoder}
import com.johnsnowlabs.ml.tensorflow.sign.{ModelSignatureConstants, ModelSignatureManager}
import com.johnsnowlabs.ml.tensorflow.{TensorResources, TensorflowWrapper}
import com.johnsnowlabs.ml.util.{ONNX, Openvino, TensorFlow}
import com.johnsnowlabs.nlp.annotators.common._
import org.intel.openvino.Tensor
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.JavaConverters._

/** The CamemBERT model was proposed in CamemBERT: a Tasty French Language Model by Louis Martin,
  * Benjamin Muller, Pedro Javier Ortiz Suárez, Yoann Dupont, Laurent Romary, Éric Villemonte de
  * la Clergerie, Djamé Seddah, and Benoît Sagot. It is based on Facebook’s RoBERTa model released
  * in 2019. It is a model trained on 138GB of French text.
  *
  * @param tensorflowWrapper
  *   CamemBERRT Model wrapper with TensorFlowWrapper
  * @param spp
  *   CamemBERRT SentencePiece model with SentencePieceWrapper
  * @param configProtoBytes
  *   Configuration for TensorFlow session
  */
private[johnsnowlabs] class CamemBert(
    val tensorflowWrapper: Option[TensorflowWrapper],
    val onnxWrapper: Option[OnnxWrapper],
    val openvinoWrapper: Option[OpenvinoWrapper],
    val spp: SentencePieceWrapper,
    configProtoBytes: Option[Array[Byte]] = None,
    signatures: Option[Map[String, String]] = None)
    extends Serializable {

  protected val logger: Logger = LoggerFactory.getLogger("CamemBert")
  val _tfCamemBertSignatures: Map[String, String] =
    signatures.getOrElse(ModelSignatureManager.apply())

  val detectedEngine: String =
    if (tensorflowWrapper.isDefined) TensorFlow.name
    else if (onnxWrapper.isDefined) ONNX.name
    else if (openvinoWrapper.isDefined) Openvino.name
    else TensorFlow.name
  private val onnxSessionOptions: Map[String, String] = new OnnxSession().getSessionOptions

  /** HACK: These tokens were added by fairseq but don't seem to be actually used when duplicated
    * in the actual # sentencepiece vocabulary (this is the case for '''<s>''' and '''</s>''')
    * '''<s>NOTUSED": 0''','''"<pad>": 1''', '''"</s>NOTUSED": 2''', '''"<unk>": 3'''
    */
  private val PieceIdOffset = 4
  private val SentenceStartTokenId = spp.getSppModel.pieceToId("<s>") + PieceIdOffset
  private val SentenceEndTokenId = spp.getSppModel.pieceToId("</s>") + PieceIdOffset
  private val SentencePadTokenId = spp.getSppModel.pieceToId("<pad>") + PieceIdOffset
  private val SentencePieceDelimiterId = spp.getSppModel.pieceToId("▁") + PieceIdOffset

  private def sessionWarmup(): Unit = {
    val dummyInput =
      Array(5, 54, 110, 11, 10, 15540, 215, 1280, 808, 25352, 1782, 808, 24696, 378, 17409, 9, 6)
    tag(Seq(dummyInput))
  }

  sessionWarmup()

  def tag(batch: Seq[Array[Int]]): Seq[Array[Array[Float]]] = {

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

        val inputs =
          Map("input_ids" -> tokenTensors, "attention_mask" -> maskTensors).asJava

        // TODO:  A try without a catch or finally is equivalent to putting its body in a block; no exceptions are handled.
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
        }

      case Openvino.name =>


        val batchLength = batch.length
        val (tokenTensors, maskTensors) =
          PrepareEmbeddings.prepareOvLongBatchTensors(batch, maxSentenceLength, batchLength, SentencePadTokenId)

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
            tensors = tensors,
            batch = batch,
            maxSentenceLength = maxSentenceLength,
            batchLength = batchLength)

        val runner = tensorflowWrapper.get
          .getTFSessionWithSignature(
            configProtoBytes = configProtoBytes,
            savedSignatures = signatures,
            initAllTables = false)
          .runner

        runner
          .feed(
            _tfCamemBertSignatures.getOrElse(
              ModelSignatureConstants.InputIdsV1.key,
              "missing_input_id_key"),
            tokenTensors)
          .feed(
            _tfCamemBertSignatures
              .getOrElse(ModelSignatureConstants.AttentionMaskV1.key, "missing_input_mask_key"),
            maskTensors)
          .fetch(
            _tfCamemBertSignatures
              .getOrElse(
                ModelSignatureConstants.LastHiddenStateV1.key,
                "missing_sequence_output_key"))

        val outs = runner.run().asScala
        val embeddings = TensorResources.extractFloats(outs.head)

        tokenTensors.close()
        maskTensors.close()
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
      new SentencepieceEncoder(
        spp,
        caseSensitive,
        delimiterId = SentencePieceDelimiterId,
        pieceIdOffset = 4)

    val sentenceTokenPieces = sentences.map { s =>
      val trimmedSentence = s.indexedTokens.take(maxSeqLength - 2)
      val wordpieceTokens =
        trimmedSentence.flatMap(token => encoder.encode(token)).take(maxSeqLength)
      WordpieceTokenizedSentence(wordpieceTokens)
    }
    sentenceTokenPieces
  }

}
