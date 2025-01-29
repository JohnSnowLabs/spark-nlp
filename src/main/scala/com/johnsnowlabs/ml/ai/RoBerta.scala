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

import ai.onnxruntime.{OnnxTensor, TensorInfo}
import com.johnsnowlabs.ml.ai.util.PrepareEmbeddings
import com.johnsnowlabs.ml.onnx.{OnnxSession, OnnxWrapper}
import com.johnsnowlabs.ml.openvino.OpenvinoWrapper
import com.johnsnowlabs.ml.tensorflow.sign.{ModelSignatureConstants, ModelSignatureManager}
import com.johnsnowlabs.ml.tensorflow.{TensorResources, TensorflowWrapper}
import com.johnsnowlabs.ml.util.{LinAlg, ModelArch, ONNX, Openvino, TensorFlow}
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.JavaConverters._

/** TensorFlow backend for '''RoBERTa''' and '''Longformer'''
  *
  * @param tensorflowWrapper
  *   tensorflowWrapper class
  * @param onnxWrapper
  *   Model wrapper with ONNX Wrapper
  * @param openvinoWrapper
  *   Model wrapper with OpenVINO Wrapper
  * @param sentenceStartTokenId
  *   special token id for `<s>`
  * @param sentenceEndTokenId
  *   special token id for `</s>`
  * @param configProtoBytes
  *   ProtoBytes for TensorFlow session config
  * @param signatures
  *   Model's inputs and output(s) signatures
  */
private[johnsnowlabs] class RoBerta(
    val tensorflowWrapper: Option[TensorflowWrapper],
    val onnxWrapper: Option[OnnxWrapper],
    val openvinoWrapper: Option[OpenvinoWrapper],
    sentenceStartTokenId: Int,
    sentenceEndTokenId: Int,
    padTokenId: Int,
    configProtoBytes: Option[Array[Byte]] = None,
    signatures: Option[Map[String, String]] = None,
    modelArch: String = ModelArch.wordEmbeddings)
    extends Serializable {

  protected val logger: Logger = LoggerFactory.getLogger("Roberta")
  val _tfRoBertaSignatures: Map[String, String] =
    signatures.getOrElse(ModelSignatureManager.apply())
  val detectedEngine: String =
    if (tensorflowWrapper.isDefined) TensorFlow.name
    else if (onnxWrapper.isDefined) ONNX.name
    else if (openvinoWrapper.isDefined) Openvino.name
    else TensorFlow.name
  private val onnxSessionOptions: Map[String, String] = new OnnxSession().getSessionOptions

  private def sessionWarmup(): Unit = {
    val dummyInput =
      Array(0, 7939, 18, 3279, 658, 5, 19374, 13, 5, 78, 42752, 4, 2)
    if (modelArch == ModelArch.wordEmbeddings) {
      tag(Seq(dummyInput))
    } else if (modelArch == ModelArch.sentenceEmbeddings) {
      tagSequence(Seq(dummyInput))
    }
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
        val (tokenTensors, maskTensors) = PrepareEmbeddings.prepareOvLongBatchTensors(
          batch = batch,
          maxSentenceLength = maxSentenceLength,
          batchLength = batchLength,
          sentencePadTokenId = padTokenId)

        val inferRequest = openvinoWrapper.get.getCompiledModel().create_infer_request()
        inferRequest.set_tensor("input_ids", tokenTensors)
        inferRequest.set_tensor("attention_mask", maskTensors)

        inferRequest.infer()

        val result = inferRequest.get_tensor("last_hidden_state")
        val embeddings = result.data()

        embeddings

      case _ =>
        val tensors = new TensorResources()

        val (tokenTensors, maskTensors) =
          PrepareEmbeddings.prepareBatchTensors(
            tensors = tensors,
            batch = batch,
            maxSentenceLength = maxSentenceLength,
            batchLength = batchLength,
            sentencePadTokenId = padTokenId)

        val runner = tensorflowWrapper.get
          .getTFSessionWithSignature(
            configProtoBytes = configProtoBytes,
            savedSignatures = signatures,
            initAllTables = false)
          .runner

        runner
          .feed(
            _tfRoBertaSignatures
              .getOrElse(ModelSignatureConstants.InputIds.key, "missing_input_id_key"),
            tokenTensors)
          .feed(
            _tfRoBertaSignatures
              .getOrElse(ModelSignatureConstants.AttentionMask.key, "missing_input_mask_key"),
            maskTensors)
          .fetch(
            _tfRoBertaSignatures
              .getOrElse(
                ModelSignatureConstants.LastHiddenState.key,
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

  /** @param batch
    *   batches of sentences
    * @return
    *   batches of vectors for each sentence
    */
  def tagSequence(batch: Seq[Array[Int]]): Array[Array[Float]] = {

    val maxSentenceLength = batch.map(pieceIds => pieceIds.length).max
    val batchLength = batch.length

    val embeddings = detectedEngine match {
      case ONNX.name =>
        val (runner, env) = onnxWrapper.get.getSession(onnxSessionOptions)

        val tokenTensors =
          OnnxTensor.createTensor(env, batch.map(x => x.map(x => x.toLong)).toArray)
        val attentionMask = batch
          .map(sentence => sentence.map(x => if (x == padTokenId) 0L else 1L))
          .toArray
        val maskTensors =
          OnnxTensor.createTensor(env, attentionMask)

        val inputs =
          Map("input_ids" -> tokenTensors, "attention_mask" -> maskTensors).asJava

        try {
          val results = runner.run(inputs)
          val lastHiddenState = results.get("last_hidden_state").get()
          val info = lastHiddenState.getInfo.asInstanceOf[TensorInfo]
          val tensorShape = info.getShape
          try {
            val flattenEmbeddings = results
              .get("last_hidden_state")
              .get()
              .asInstanceOf[OnnxTensor]
              .getFloatBuffer
              .array()
            tokenTensors.close()
            maskTensors.close()
            val embeddings = LinAlg.avgPooling(flattenEmbeddings, attentionMask, tensorShape)
            val normalizedEmbeddings = LinAlg.l2Normalize(embeddings)
            LinAlg.denseMatrixToArray(normalizedEmbeddings)
          } finally if (results != null) results.close()
        } catch {
          case e: Exception =>
            // Log the exception as a warning
            logger.warn("Exception: ", e)
            // Rethrow the exception to propagate it further
            throw e
        }

      case Openvino.name =>
        val shape = Array(batchLength, maxSentenceLength)
        val tokenTensors =
          new org.intel.openvino.Tensor(shape, batch.flatMap(x => x.map(xx => xx.toLong)).toArray)

        val attentionMask = batch
          .map(sentence => sentence.map(x => if (x == padTokenId) 0L else 1L))
          .toArray

        val maskTensors = new org.intel.openvino.Tensor(shape, attentionMask.flatten)
        val inferRequest = openvinoWrapper.get.getCompiledModel().create_infer_request()
        inferRequest.set_tensor("input_ids", tokenTensors)
        inferRequest.set_tensor("attention_mask", maskTensors)

        inferRequest.infer()

        val lastHiddenState = inferRequest
          .get_tensor("last_hidden_state")
        val tensorShape = lastHiddenState.get_shape().map(_.toLong)
        val flattenEmbeddings = lastHiddenState
          .data()
        val embeddings = LinAlg.avgPooling(flattenEmbeddings, attentionMask, tensorShape)
        val normalizedEmbeddings = LinAlg.l2Normalize(embeddings)
        LinAlg.denseMatrixToArray(normalizedEmbeddings)

      case _ =>
        val tensors = new TensorResources()

        val (tokenTensors, maskTensors) =
          PrepareEmbeddings.prepareBatchTensors(
            tensors = tensors,
            batch = batch,
            maxSentenceLength = maxSentenceLength,
            batchLength = batchLength,
            sentencePadTokenId = padTokenId)

        val runner = tensorflowWrapper.get
          .getTFSessionWithSignature(
            configProtoBytes = configProtoBytes,
            savedSignatures = signatures,
            initAllTables = false)
          .runner

        runner
          .feed(
            _tfRoBertaSignatures
              .getOrElse(ModelSignatureConstants.InputIds.key, "missing_input_id_key"),
            tokenTensors)
          .feed(
            _tfRoBertaSignatures
              .getOrElse(ModelSignatureConstants.AttentionMask.key, "missing_input_mask_key"),
            maskTensors)
          .fetch(_tfRoBertaSignatures
            .getOrElse(ModelSignatureConstants.PoolerOutput.key, "missing_pooled_output_key"))

        val outs = runner.run().asScala
        val embeddings = TensorResources.extractFloats(outs.head)

        tokenTensors.close()
        maskTensors.close()
        tensors.clearSession(outs)
        tensors.clearTensors()

        val dim = embeddings.length / batchLength
        embeddings.grouped(dim).toArray

    }

    embeddings

  }

  def predict(
      sentences: Seq[WordpieceTokenizedSentence],
      originalTokenSentences: Seq[TokenizedSentence],
      batchSize: Int,
      maxSentenceLength: Int,
      caseSensitive: Boolean): Seq[WordpieceEmbeddingsSentence] = {

    /*Run embeddings calculation by batches*/
    sentences.zipWithIndex
      .grouped(batchSize)
      .flatMap { batch =>
        val encoded = PrepareEmbeddings.prepareBatchInputsWithPadding(
          batch,
          maxSentenceLength,
          sentenceStartTokenId,
          sentenceEndTokenId,
          padTokenId)
        val vectors = tag(encoded)

        /*Combine tokens and calculated embeddings*/
        batch.zip(vectors).map { case (sentence, tokenVectors) =>
          val tokenLength = sentence._1.tokens.length

          /*All wordpiece embeddings*/
          val tokenEmbeddings = tokenVectors.slice(1, tokenLength + 1)
          val originalIndexedTokens = originalTokenSentences(sentence._2)

          /*Word-level and span-level alignment with Tokenizer
        https://github.com/google-research/bert#tokenization

        ### Input
        orig_tokens = ["John", "Johanson", "'s",  "house"]
        labels      = ["NNP",  "NNP",      "POS", "NN"]

        # bert_tokens == ["[CLS]", "john", "johan", "##son", "'", "s", "house", "[SEP]"]
        # orig_to_tok_map == [1, 2, 4, 6]*/

          val tokensWithEmbeddings =
            sentence._1.tokens.zip(tokenEmbeddings).flatMap { case (token, tokenEmbedding) =>
              val tokenWithEmbeddings = TokenPieceEmbeddings(token, tokenEmbedding)
              val originalTokensWithEmbeddings = originalIndexedTokens.indexedTokens
                .find(p => p.begin == tokenWithEmbeddings.begin)
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

  def predictSequence(
      tokens: Seq[WordpieceTokenizedSentence],
      sentences: Seq[Sentence],
      batchSize: Int,
      maxSentenceLength: Int): Seq[Annotation] = {

    /*Run embeddings calculation by batches*/
    tokens
      .zip(sentences)
      .zipWithIndex
      .grouped(batchSize)
      .flatMap { batch =>
        val tokensBatch = batch.map(x => (x._1._1, x._2))
        val sentencesBatch = batch.map(x => x._1._2)
        val encoded = PrepareEmbeddings.prepareBatchInputsWithPadding(
          tokensBatch,
          maxSentenceLength,
          sentenceStartTokenId,
          sentenceEndTokenId,
          padTokenId)
        val embeddings = tagSequence(encoded)

        sentencesBatch.zip(embeddings).map { case (sentence, vectors) =>
          Annotation(
            annotatorType = AnnotatorType.SENTENCE_EMBEDDINGS,
            begin = sentence.start,
            end = sentence.end,
            result = sentence.content,
            metadata = Map(
              "sentence" -> sentence.index.toString,
              "token" -> sentence.content,
              "pieceId" -> "-1",
              "isWordStart" -> "true"),
            embeddings = vectors)
        }
      }
      .toSeq
  }

}
