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

import com.johnsnowlabs.ml.ai.util.PrepareEmbeddings
import com.johnsnowlabs.ml.tensorflow.sign.{ModelSignatureConstants, ModelSignatureManager}
import com.johnsnowlabs.ml.tensorflow.{TensorResources, TensorflowWrapper}
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}

import scala.collection.JavaConverters._

/** BERT (Bidirectional Encoder Representations from Transformers) provides dense vector
  * representations for natural language by using a deep, pre-trained neural network with the
  * Transformer architecture
  *
  * See
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/BertEmbeddingsTestSpec.scala]]
  * for further reference on how to use this API. Sources:
  *
  * @param tensorflowWrapper
  *   Bert Model wrapper with TensorFlow Wrapper
  * @param sentenceStartTokenId
  *   Id of sentence start Token
  * @param sentenceEndTokenId
  *   Id of sentence end Token.
  * @param configProtoBytes
  *   Configuration for TensorFlow session
  *
  * Paper: [[https://arxiv.org/abs/1810.04805]]
  *
  * Source: [[https://github.com/google-research/bert]]
  */
private[johnsnowlabs] class Bert(
    val tensorflowWrapper: TensorflowWrapper,
    sentenceStartTokenId: Int,
    sentenceEndTokenId: Int,
    configProtoBytes: Option[Array[Byte]] = None,
    signatures: Option[Map[String, String]] = None)
    extends Serializable {

  val _tfBertSignatures: Map[String, String] = signatures.getOrElse(ModelSignatureManager.apply())

  def tag(batch: Seq[Array[Int]]): Seq[Array[Array[Float]]] = {

    val maxSentenceLength = batch.map(pieceIds => pieceIds.length).max
    val batchLength = batch.length

    val tensors = new TensorResources()

    val (tokenTensors, maskTensors, segmentTensors) =
      PrepareEmbeddings.prepareBatchTensorsWithSegment(
        tensors = tensors,
        batch = batch,
        maxSentenceLength = maxSentenceLength,
        batchLength = batchLength)

    val runner = tensorflowWrapper
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
        .getOrElse(ModelSignatureConstants.LastHiddenStateV1.key, "missing_sequence_output_key"))

    val outs = runner.run().asScala
    val embeddings = TensorResources.extractFloats(outs.head)

    tokenTensors.close()
    maskTensors.close()
    segmentTensors.close()
    tensors.clearSession(outs)
    tensors.clearTensors()

    PrepareEmbeddings.prepareBatchWordEmbeddings(
      batch,
      embeddings,
      maxSentenceLength,
      batchLength)

  }

  def tagSequence(batch: Seq[Array[Int]]): Array[Array[Float]] = {

    val maxSentenceLength = batch.map(pieceIds => pieceIds.length).max
    val batchLength = batch.length

    val tensors = new TensorResources()

    val (tokenTensors, maskTensors, segmentTensors) =
      PrepareEmbeddings.prepareBatchTensorsWithSegment(
        tensors = tensors,
        batch = batch,
        maxSentenceLength = maxSentenceLength,
        batchLength = batchLength)

    val runner = tensorflowWrapper
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

    val dim = embeddings.length / batchLength
    embeddings.grouped(dim).toArray

  }

  def tagSequenceSBert(batch: Seq[Array[Int]]): Array[Array[Float]] = {

    val tensors = new TensorResources()

    val maxSentenceLength = batch.map(x => x.length).max
    val batchLength = batch.length

    val tokenBuffers = tensors.createLongBuffer(batchLength * maxSentenceLength)
    val maskBuffers = tensors.createLongBuffer(batchLength * maxSentenceLength)
    val segmentBuffers = tensors.createLongBuffer(batchLength * maxSentenceLength)

    val shape = Array(batchLength.toLong, maxSentenceLength)

    batch.zipWithIndex.foreach { case (sentence, idx) =>
      val offset = idx * maxSentenceLength
      tokenBuffers.offset(offset).write(sentence.map(_.toLong))
      maskBuffers.offset(offset).write(sentence.map(x => if (x == 0L) 0L else 1L))
      segmentBuffers.offset(offset).write(Array.fill(maxSentenceLength)(0L))
    }

    val runner = tensorflowWrapper
      .getTFSessionWithSignature(
        configProtoBytes = configProtoBytes,
        savedSignatures = signatures,
        initAllTables = false)
      .runner

    val tokenTensors = tensors.createLongBufferTensor(shape, tokenBuffers)
    val maskTensors = tensors.createLongBufferTensor(shape, maskBuffers)
    val segmentTensors = tensors.createLongBufferTensor(shape, segmentBuffers)

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

  def predictSequence(
      tokens: Seq[WordpieceTokenizedSentence],
      sentences: Seq[Sentence],
      batchSize: Int,
      maxSentenceLength: Int,
      isLong: Boolean = false): Seq[Annotation] = {

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

        val embeddings = if (isLong) {
          tagSequenceSBert(encoded)
        } else {
          tagSequence(encoded)
        }

        sentencesBatch.zip(embeddings).map { case (sentence, vectors) =>
          val metadata = Map(
            "sentence" -> sentence.index.toString,
            "token" -> sentence.content,
            "pieceId" -> "-1",
            "isWordStart" -> "true")
          val finalMetadata = if (sentence.metadata.isDefined) {
            sentence.metadata.getOrElse(Map.empty) ++ metadata
          } else {
            metadata
          }
          Annotation(
            annotatorType = AnnotatorType.SENTENCE_EMBEDDINGS,
            begin = sentence.start,
            end = sentence.end,
            result = sentence.content,
            metadata = finalMetadata,
            embeddings = vectors)
        }
      }
      .toSeq
  }

}
