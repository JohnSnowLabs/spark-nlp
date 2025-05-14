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
import com.johnsnowlabs.ml.tensorflow.sign.{ModelSignatureConstants, ModelSignatureManager}
import com.johnsnowlabs.ml.tensorflow.{TensorResources, TensorflowWrapper}
import com.johnsnowlabs.ml.util.{ModelArch, ONNX, Openvino, TensorFlow}
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.JavaConverters._

/** The DistilBERT model was proposed in the paper '''DistilBERT, a distilled version of BERT:
  * smaller, faster, cheaper and lighter''' [[https://arxiv.org/abs/1910.01108]]. DistilBERT is a
  * small, fast, cheap and light Transformer model trained by distilling BERT base. It has 40%
  * less parameters than `bert-base-uncased`, runs 60% faster while preserving over 95% of BERT's
  * performances as measured on the GLUE language understanding benchmark.
  *
  * The abstract from the paper is the following:
  *
  * As Transfer Learning from large-scale pre-trained models becomes more prevalent in Natural
  * Language Processing (NLP), operating these large models in on-the-edge and/or under
  * constrained computational training or inference budgets remains challenging. In this work, we
  * propose a method to pre-train a smaller general-purpose language representation model, called
  * DistilBERT, which can then be fine-tuned with good performances on a wide range of tasks like
  * its larger counterparts. While most prior work investigated the use of distillation for
  * building task-specific models, we leverage knowledge distillation during the pretraining phase
  * and show that it is possible to reduce the size of a BERT model by 40%, while retaining 97% of
  * its language understanding capabilities and being 60% faster. To leverage the inductive biases
  * learned by larger models during pretraining, we introduce a triple loss combining language
  * modeling, distillation and cosine-distance losses. Our smaller, faster and lighter model is
  * cheaper to pre-train and we demonstrate its capabilities for on-device computations in a
  * proof-of-concept experiment and a comparative on-device study.
  *
  * Tips:
  *
  *   - DistilBERT doesn't have :obj:`token_type_ids`, you don't need to indicate which token
  *     belongs to which segment. Just separate your segments with the separation token
  *     :obj:`tokenizer.sep_token` (or :obj:`[SEP]`).
  *
  *   - DistilBERT doesn't have options to select the input positions (:obj:`position_ids` input).
  *     This could be added if necessary though, just let us know if you need this option.
  *
  * @param tensorflowWrapper
  *   Bert Model wrapper with TensorFlow Wrapper
  * @param sentenceStartTokenId
  *   Id of sentence start Token
  * @param sentenceEndTokenId
  *   Id of sentence end Token.
  * @param configProtoBytes
  *   Configuration for TensorFlow session
  */
private[johnsnowlabs] class DistilBert(
    val tensorflowWrapper: Option[TensorflowWrapper],
    val onnxWrapper: Option[OnnxWrapper],
    val openvinoWrapper: Option[OpenvinoWrapper],
    sentenceStartTokenId: Int,
    sentenceEndTokenId: Int,
    configProtoBytes: Option[Array[Byte]] = None,
    signatures: Option[Map[String, String]] = None,
    modelArch: String = ModelArch.wordEmbeddings)
    extends Serializable {

  protected val logger: Logger = LoggerFactory.getLogger("DistilBert")
  val _tfBertSignatures: Map[String, String] = signatures.getOrElse(ModelSignatureManager.apply())
  val detectedEngine: String =
    if (tensorflowWrapper.isDefined) TensorFlow.name
    else if (onnxWrapper.isDefined) ONNX.name
    else if (openvinoWrapper.isDefined) Openvino.name
    else TensorFlow.name
  private val onnxSessionOptions: Map[String, String] = new OnnxSession().getSessionOptions

  private def sessionWarmup(): Unit = {
    val dummyInput =
      Array(101, 2292, 1005, 1055, 4010, 6279, 1996, 5219, 2005, 1996, 2034, 28937, 1012, 102)
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
            _tfBertSignatures.getOrElse(
              ModelSignatureConstants.InputIds.key,
              "missing_input_id_key"),
            tokenTensors)
          .feed(
            _tfBertSignatures
              .getOrElse(ModelSignatureConstants.AttentionMask.key, "missing_input_mask_key"),
            maskTensors)
          .fetch(
            _tfBertSignatures
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

  /** @param batch
    *   batches of sentences
    * @return
    *   batches of vectors for each sentence
    */
  def tagSequence(batch: Seq[Array[Int]]): Array[Array[Float]] = {

    val maxSentenceLength = batch.map(pieceIds => pieceIds.length).max
    val batchLength = batch.length

    val tensors = new TensorResources()

    val (tokenTensors, maskTensors) =
      PrepareEmbeddings.prepareBatchTensors(
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
        _tfBertSignatures.getOrElse(ModelSignatureConstants.InputIds.key, "missing_input_id_key"),
        tokenTensors)
      .feed(
        _tfBertSignatures
          .getOrElse(ModelSignatureConstants.AttentionMask.key, "missing_input_mask_key"),
        maskTensors)
      .fetch(_tfBertSignatures
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
          sentenceEndTokenId)
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
