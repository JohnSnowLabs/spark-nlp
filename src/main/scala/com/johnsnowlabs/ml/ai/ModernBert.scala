/*
 * Copyright 2017-2025 John Snow Labs
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
import com.johnsnowlabs.ml.ai.util.Generation.GenerationConfig
import com.johnsnowlabs.ml.ai.util.PrepareEmbeddings
import com.johnsnowlabs.ml.onnx.{OnnxSession, OnnxWrapper}
import com.johnsnowlabs.ml.openvino.OpenvinoWrapper
import com.johnsnowlabs.ml.tensorflow.sign.{ModelSignatureConstants, ModelSignatureManager}
import com.johnsnowlabs.ml.tensorflow.{TensorResources, TensorflowWrapper}
import com.johnsnowlabs.ml.util.{ModelArch, ONNX, Openvino, TensorFlow}
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.annotators.tokenizer.bpe.{
  BpeTokenizer,
  ModernBertTokenizer,
  SpecialTokens
}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import org.intel.openvino.Tensor
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.JavaConverters._

/** ModernBERT (Modern Bidirectional Encoder Representations from Transformers) provides dense
  * vector representations for natural language by using a modernized deep, pre-trained neural
  * network with the Transformer architecture.
  *
  * See
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/ModernBertEmbeddingsTestSpec.scala]]
  * for further reference on how to use this API. Sources:
  *
  * @param tensorflowWrapper
  *   ModernBert Model wrapper with TensorFlow Wrapper
  * @param onnxWrapper
  *   ModernBert Model wrapper with ONNX Wrapper
  * @param openvinoWrapper
  *   ModernBert Model wrapper with OpenVINO Wrapper
  * @param sentenceStartTokenId
  *   Id of sentence start Token
  * @param sentenceEndTokenId
  *   Id of sentence end Token.
  * @param configProtoBytes
  *   Configuration for TensorFlow session
  *
  * Paper: [[https://arxiv.org/abs/2412.13663]]
  *
  * Source: [[https://huggingface.co/answerdotai/ModernBERT-base]]
  */
private[johnsnowlabs] class ModernBert(
    val tensorflowWrapper: Option[TensorflowWrapper],
    val onnxWrapper: Option[OnnxWrapper],
    val openvinoWrapper: Option[OpenvinoWrapper],
    sentenceStartTokenId: Int,
    sentenceEndTokenId: Int,
    merges: Map[(String, String), Int],
    vocabulary: Map[String, Int],
    addedTokens: Map[String, Int],
    configProtoBytes: Option[Array[Byte]] = None,
    signatures: Option[Map[String, String]] = None,
    modelArch: String = ModelArch.wordEmbeddings,
    isSBert: Boolean = false)
    extends Serializable {

  protected val logger: Logger = LoggerFactory.getLogger("ModernBert")
  val _tfBertSignatures: Map[String, String] = signatures.getOrElse(ModelSignatureManager.apply())
  val detectedEngine: String =
    if (tensorflowWrapper.isDefined) TensorFlow.name
    else if (onnxWrapper.isDefined) ONNX.name
    else if (openvinoWrapper.isDefined) Openvino.name
    else TensorFlow.name
  private val onnxSessionOptions: Map[String, String] = new OnnxSession().getSessionOptions

  // For ModernBERT, we use [CLS] and [SEP] tokens directly
  val reversedVocabulary: Map[Int, String] = vocabulary.map(_.swap)
  val specialTokens: SpecialTokens = SpecialTokens(
    vocabulary,
    startTokenString = "[CLS]",
    endTokenString = "[SEP]",
    unkTokenString = "[UNK]",
    maskTokenString = "[MASK]",
    padTokenString = "[PAD]",
    additionalStrings = addedTokens.keys.toArray)

  val bpeTokenizer: ModernBertTokenizer =
    new ModernBertTokenizer(merges, vocabulary, specialTokens)
  private def sessionWarmup(): Unit = {
    val dummyInput =
      Array(101, 2292, 1005, 1055, 4010, 6279, 1996, 5219, 2005, 1996, 2034, 28937, 1012, 102)
    if (modelArch == ModelArch.wordEmbeddings) {
      tag(Seq(dummyInput))
    } else if (modelArch == ModelArch.sentenceEmbeddings) {
      if (isSBert)
        tagSequenceSBert(Seq(dummyInput))
      else
        tagSequence(Seq(dummyInput))
    }
  }

  sessionWarmup()

  def tokenizeWithAlignment(
      tokens: Seq[TokenizedSentence],
      maxSentenceLength: Int,
      caseSensitive: Boolean): Seq[WordpieceTokenizedSentence] = {

    tokens.map { tokenIndex =>
      // filter empty and only whitespace tokens
      val bertTokens =
        tokenIndex.indexedTokens.filter(x => x.token.nonEmpty && !x.token.equals(" ")).map {
          token =>
            val content = if (caseSensitive) token.token else token.token.toLowerCase()
            IndexedToken(content, token.begin, token.end)
        }
      val wordpieceTokens =
        bertTokens.flatMap(token => bpeTokenizer.encode(token)).take(maxSentenceLength)
      WordpieceTokenizedSentence(wordpieceTokens)
    }
  }

  def predict(
      tokenizedSentences: Seq[WordpieceTokenizedSentence],
      originalTokens: Seq[TokenizedSentence],
      batchSize: Int,
      maxSentenceLength: Int,
      caseSensitive: Boolean): Seq[WordpieceEmbeddingsSentence] = {

    /*Run embeddings calculation by batches*/
    tokenizedSentences.zipWithIndex
      .grouped(batchSize)
      .flatMap { batch =>
        val encoded = PrepareEmbeddings.prepareBatchInputsWithPadding(
          batch,
          maxSentenceLength,
          sentenceStartTokenId,
          sentenceEndTokenId)
        val vectors = tag(encoded)

        /*Word-level and span-level alignment*/
        batch.zip(vectors).map { case (sentence, tokenVectors) =>
          val tokenizedSentence = sentence._1
          val originalSentence = originalTokens(sentence._2)
          val tokenLength = tokenizedSentence.tokens.length

          /*All wordpiece embeddings*/
          val tokenEmbeddings = tokenVectors.slice(1, tokenLength + 1)

          val tokensWithEmbeddings =
            tokenizedSentence.tokens.zip(tokenEmbeddings).flatMap {
              case (token, tokenEmbedding) =>
                val tokenWithEmbeddings = TokenPieceEmbeddings(token, tokenEmbedding)
                val originalTokensWithEmbeddings = originalSentence.indexedTokens
                  .find(p =>
                    p.begin == tokenWithEmbeddings.begin && tokenWithEmbeddings.isWordStart)
                  .map { originalToken =>
                    val originalTokenWithEmbedding = TokenPieceEmbeddings(
                      TokenPiece(
                        wordpiece = tokenWithEmbeddings.wordpiece,
                        token =
                          if (caseSensitive) originalToken.token
                          else originalToken.token.toLowerCase(),
                        pieceId = tokenWithEmbeddings.pieceId,
                        isWordStart = tokenWithEmbeddings.isWordStart,
                        begin = originalToken.begin,
                        end = originalToken.end),
                      tokenEmbedding)
                    originalTokenWithEmbedding
                  }
                originalTokensWithEmbeddings
            }

          WordpieceEmbeddingsSentence(tokensWithEmbeddings, originalSentence.sentenceIndex)
        }
      }
      .toSeq
  }

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

        val segmentTensors =
          OnnxTensor.createTensor(env, batch.map(x => Array.fill(maxSentenceLength)(0L)).toArray)

        val inputs =
          Map(
            "input_ids" -> tokenTensors,
            "attention_mask" -> maskTensors,
            "token_type_ids" -> segmentTensors).asJava

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
          tokenTensors.close()
          maskTensors.close()
          segmentTensors.close()
        }
      case Openvino.name =>
        val shape = Array(batchLength, maxSentenceLength)
        val (tokenTensors, maskTensors) =
          PrepareEmbeddings.prepareOvLongBatchTensors(batch, maxSentenceLength, batchLength)
        val segmentTensors = new Tensor(shape, Array.fill(batchLength * maxSentenceLength)(0L))

        val inferRequest = openvinoWrapper.get.getCompiledModel().create_infer_request()
        inferRequest.set_tensor("input_ids", tokenTensors)
        inferRequest.set_tensor("attention_mask", maskTensors)
        inferRequest.set_tensor("token_type_ids", segmentTensors)

        inferRequest.infer()

        val result = inferRequest.get_tensor("last_hidden_state")
        val embeddings = result.data()

        embeddings
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
            _tfBertSignatures.getOrElse(
              ModelSignatureConstants.InputIdsV1.key,
              "missing_input_id_key"),
            tokenTensors)
          .feed(
            _tfBertSignatures
              .getOrElse(ModelSignatureConstants.AttentionMaskV1.key, "missing_input_mask_key"),
            maskTensors)
          .feed(
            _tfBertSignatures
              .getOrElse(ModelSignatureConstants.TokenTypeIdsV1.key, "missing_segment_ids_key"),
            segmentTensors)
          .fetch(
            _tfBertSignatures
              .getOrElse(
                ModelSignatureConstants.LastHiddenStateV1.key,
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

  def tagSequence(batch: Seq[Array[Int]]): Array[Array[Float]] = {
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
          Map(
            "input_ids" -> tokenTensors,
            "attention_mask" -> maskTensors,
            "token_type_ids" -> segmentTensors).asJava

        try {
          val results = runner.run(inputs)
          try {
            val embeddings = results
              .get("last_hidden_state")
              .get()
              .asInstanceOf[OnnxTensor]
              .getFloatBuffer
              .array()
            tokenTensors.close()
            maskTensors.close()
            segmentTensors.close()
            //    runner.close()
            //    env.close()
            //
            embeddings
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
        val (tokenTensors, maskTensors) =
          PrepareEmbeddings.prepareOvLongBatchTensors(batch, maxSentenceLength, batchLength)
        val segmentTensors = new Tensor(shape, Array.fill(batchLength * maxSentenceLength)(0L))

        val inferRequest = openvinoWrapper.get.getCompiledModel().create_infer_request()
        inferRequest.set_tensor("input_ids", tokenTensors)
        inferRequest.set_tensor("attention_mask", maskTensors)
        inferRequest.set_tensor("token_type_ids", segmentTensors)

        inferRequest.infer()

        val result = inferRequest.get_tensor("last_hidden_state")
        val embeddings = result.data()
        embeddings
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
            _tfBertSignatures.getOrElse(
              ModelSignatureConstants.InputIdsV1.key,
              "missing_input_id_key"),
            tokenTensors)
          .feed(
            _tfBertSignatures
              .getOrElse(ModelSignatureConstants.AttentionMaskV1.key, "missing_input_mask_key"),
            maskTensors)
          .feed(
            _tfBertSignatures
              .getOrElse(ModelSignatureConstants.TokenTypeIdsV1.key, "missing_segment_ids_key"),
            segmentTensors)
          .fetch(_tfBertSignatures
            .getOrElse(ModelSignatureConstants.PoolerOutput.key, "missing_pooled_output_key"))

        val outs = runner.run().asScala
        val embeddings = TensorResources.extractFloats(outs.head)

        tokenTensors.close()
        maskTensors.close()
        segmentTensors.close()
        tensors.clearSession(outs)
        tensors.clearTensors()

        embeddings
    }

    val dim = embeddings.length / batchLength
    embeddings.grouped(dim).toArray

  }

  def tagSequenceSBert(batch: Seq[Array[Int]]): Array[Array[Float]] = {
    val maxSentenceLength = batch.map(pieceIds => pieceIds.length).max
    val batchLength = batch.length

    val embeddings = detectedEngine match {
      case ONNX.name =>
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
          Map(
            "input_ids" -> tokenTensors,
            "attention_mask" -> maskTensors,
            "token_type_ids" -> segmentTensors).asJava

        val results = runner.run(inputs)
        val embeddings = results
          .get("last_hidden_state")
          .get()
          .asInstanceOf[OnnxTensor]
          .getFloatBuffer
          .array()

        tokenTensors.close()
        maskTensors.close()
        segmentTensors.close()
        results.close()

        embeddings
      case Openvino.name =>
        val shape = Array(batchLength, maxSentenceLength)
        val (tokenTensors, maskTensors) =
          PrepareEmbeddings.prepareOvLongBatchTensors(batch, maxSentenceLength, batchLength)
        val segmentTensors = new Tensor(shape, Array.fill(batchLength * maxSentenceLength)(0L))

        val inferRequest = openvinoWrapper.get.getCompiledModel().create_infer_request()
        inferRequest.set_tensor("input_ids", tokenTensors)
        inferRequest.set_tensor("attention_mask", maskTensors)
        inferRequest.set_tensor("token_type_ids", segmentTensors)

        inferRequest.infer()

        val result = inferRequest.get_tensor("last_hidden_state")
        val embeddings = result.data()
        embeddings
      case _ =>
        val tensors = new TensorResources()

        val batchLength = batch.length
        val tokenBuffers = tensors.createLongBuffer(batchLength * maxSentenceLength)
        val maskBuffers = tensors.createLongBuffer(batchLength * maxSentenceLength)
        val segmentBuffers = tensors.createLongBuffer(batchLength * maxSentenceLength)

        val shape = Array(batch.length.toLong, maxSentenceLength.toLong)

        batch.zipWithIndex
          .foreach { case (sentence, idx) =>
            val offset = idx * maxSentenceLength
            tokenBuffers.offset(offset).write(sentence.map(_.toLong))
            maskBuffers.offset(offset).write(sentence.map(x => if (x == 0L) 0L else 1L))
            segmentBuffers.offset(offset).write(Array.fill(maxSentenceLength)(0L))
          }

        val tokenTensors = tensors.createLongBufferTensor(shape, tokenBuffers)
        val maskTensors = tensors.createLongBufferTensor(shape, maskBuffers)
        val segmentTensors = tensors.createLongBufferTensor(shape, segmentBuffers)

        val runner = tensorflowWrapper.get
          .getTFSessionWithSignature(
            configProtoBytes = configProtoBytes,
            savedSignatures = signatures,
            initAllTables = false)
          .runner

        runner
          .feed(
            _tfBertSignatures.getOrElse(
              ModelSignatureConstants.InputIdsV1.key,
              "missing_input_id_key"),
            tokenTensors)
          .feed(
            _tfBertSignatures
              .getOrElse(ModelSignatureConstants.AttentionMaskV1.key, "missing_input_mask_key"),
            maskTensors)
          .feed(
            _tfBertSignatures
              .getOrElse(ModelSignatureConstants.TokenTypeIdsV1.key, "missing_segment_ids_key"),
            segmentTensors)
          .fetch(
            _tfBertSignatures
              .getOrElse(
                ModelSignatureConstants.LastHiddenStateV1.key,
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

    val dim = embeddings.length / (batchLength * maxSentenceLength)
    embeddings.grouped(dim).toArray
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
          sentenceEndTokenId)
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
