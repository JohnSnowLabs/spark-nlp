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
import com.johnsnowlabs.ml.tensorflow.sentencepiece.{SentencePieceWrapper, SentencepieceEncoder}
import com.johnsnowlabs.ml.tensorflow.sign.{ModelSignatureConstants, ModelSignatureManager}
import com.johnsnowlabs.ml.tensorflow.{TensorResources, TensorflowWrapper}
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}

import scala.collection.JavaConverters._

/** Sentence-level embeddings using XLM-RoBERTa. The XLM-RoBERTa model was proposed in
  * '''Unsupervised Cross-lingual Representation Learning at Scale'''
  * [[https://arxiv.org/abs/1911.02116]] by Alexis Conneau, Kartikay Khandelwal, Naman Goyal,
  * Vishrav Chaudhary, Guillaume Wenzek, Francisco GuzmÃ¡n, Edouard Grave, Myle Ott, Luke
  * Zettlemoyer and Veselin Stoyanov. It is based on Facebook's RoBERTa model released in 2019. It
  * is a large multi-lingual language model, trained on 2.5TB of filtered CommonCrawl data.
  *
  * The abstract from the paper is the following:
  *
  * This paper shows that pretraining multilingual language models at scale leads to significant
  * performance gains for a wide range of cross-lingual transfer tasks. We train a
  * Transformer-based masked language model on one hundred languages, using more than two
  * terabytes of filtered CommonCrawl data. Our model, dubbed XLM-R, significantly outperforms
  * multilingual BERT (mBERT) on a variety of cross-lingual benchmarks, including +13.8% average
  * accuracy on XNLI, +12.3% average F1 score on MLQA, and +2.1% average F1 score on NER. XLM-R
  * performs particularly well on low-resource languages, improving 11.8% in XNLI accuracy for
  * Swahili and 9.2% for Urdu over the previous XLM model. We also present a detailed empirical
  * evaluation of the key factors that are required to achieve these gains, including the
  * trade-offs between (1) positive transfer and capacity dilution and (2) the performance of high
  * and low resource languages at scale. Finally, we show, for the first time, the possibility of
  * multilingual modeling without sacrificing per-language performance; XLM-Ris very competitive
  * with strong monolingual models on the GLUE and XNLI benchmarks. We will make XLM-R code, data,
  * and models publicly available.
  *
  * Tips:
  *
  *   - XLM-RoBERTa is a multilingual model trained on 100 different languages. Unlike some XLM
  *     multilingual models, it does not require '''lang''' parameter to understand which language
  *     is used, and should be able to determine the correct language from the input ids.
  *   - This implementation is the same as RoBERTa. Refer to the
  *     [[com.johnsnowlabs.nlp.embeddings.RoBertaEmbeddings]] for usage examples as well as the
  *     information relative to the inputs and outputs.
  *
  * @param tensorflowWrapper
  *   XlmRoberta Model wrapper with TensorFlowWrapper
  * @param spp
  *   XlmRoberta SentencePiece model with SentencePieceWrapper
  * @param caseSensitive
  *   Whether or not the tokenizer should be lowercase
  * @param configProtoBytes
  *   Configuration for TensorFlow session
  * @param signatures
  *   Model's inputs and output(s) signatures
  */
private[johnsnowlabs] class XlmRoberta(
    val tensorflowWrapper: TensorflowWrapper,
    val spp: SentencePieceWrapper,
    caseSensitive: Boolean = true,
    configProtoBytes: Option[Array[Byte]] = None,
    signatures: Option[Map[String, String]] = None)
    extends Serializable {

  val _tfRoBertaSignatures: Map[String, String] =
    signatures.getOrElse(ModelSignatureManager.apply())

  private val SentenceStartTokenId = 0
  private val SentenceEndTokenId = 2
  private val SentencePadTokenId = 1
  private val SentencePieceDelimiterId = spp.getSppModel.pieceToId("▁")

  def tag(batch: Seq[Array[Int]]): Seq[Array[Array[Float]]] = {

    val maxSentenceLength = batch.map(pieceIds => pieceIds.length).max
    val batchLength = batch.length

    val tensors = new TensorResources()

    val (tokenTensors, maskTensors) =
      PrepareEmbeddings.prepareBatchTensors(
        tensors = tensors,
        batch = batch,
        maxSentenceLength = maxSentenceLength,
        batchLength = batchLength,
        sentencePadTokenId = SentencePadTokenId)

    val runner = tensorflowWrapper
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
        .getOrElse(ModelSignatureConstants.LastHiddenState.key, "missing_sequence_output_key"))

    val outs = runner.run().asScala
    val embeddings = TensorResources.extractFloats(outs.head)

    tokenTensors.close()
    maskTensors.close()
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

    val (tokenTensors, maskTensors) =
      PrepareEmbeddings.prepareBatchTensors(
        tensors = tensors,
        batch = batch,
        maxSentenceLength = maxSentenceLength,
        batchLength = batchLength,
        sentencePadTokenId = SentencePadTokenId)

    val runner = tensorflowWrapper
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

  def predict(
      tokenizedSentences: Seq[TokenizedSentence],
      batchSize: Int,
      maxSentenceLength: Int): Seq[WordpieceEmbeddingsSentence] = {

    val wordPieceTokenizedSentences = tokenizeWithAlignment(tokenizedSentences, maxSentenceLength)
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

  def predictSequence(
      sentences: Seq[Sentence],
      batchSize: Int,
      maxSentenceLength: Int): Seq[Annotation] = {

    val wordPieceTokenizedSentences = sentences
      .grouped(batchSize)
      .flatMap { batch =>
        tokenizeSentence(batch, maxSentenceLength)
      }
      .toSeq

    /*Run embeddings calculation by batches*/
    wordPieceTokenizedSentences
      .zip(sentences)
      .zipWithIndex
      .grouped(batchSize)
      .flatMap { batch =>
        val tokensBatch = batch.map(x => (x._1._1, x._2))
        val sentencesBatch = batch.map(x => x._1._2)
        val encoded = PrepareEmbeddings.prepareBatchInputsWithPadding(
          tokensBatch,
          maxSentenceLength,
          SentenceStartTokenId,
          SentenceEndTokenId,
          SentencePadTokenId)
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

  def tokenizeWithAlignment(
      sentences: Seq[TokenizedSentence],
      maxSeqLength: Int): Seq[WordpieceTokenizedSentence] = {
    val encoder =
      new SentencepieceEncoder(spp, caseSensitive, SentencePieceDelimiterId, pieceIdOffset = 1)

    val sentenceTokenPieces = sentences.map { s =>
      val trimmedSentence = s.indexedTokens.take(maxSeqLength - 2)
      val wordpieceTokens =
        trimmedSentence.flatMap(token => encoder.encode(token)).take(maxSeqLength)
      WordpieceTokenizedSentence(wordpieceTokens)
    }
    sentenceTokenPieces
  }

  def tokenizeSentence(
      sentences: Seq[Sentence],
      maxSeqLength: Int): Seq[WordpieceTokenizedSentence] = {
    val encoder =
      new SentencepieceEncoder(spp, caseSensitive, SentencePieceDelimiterId, pieceIdOffset = 1)

    val sentenceTokenPieces = sentences.map { s =>
      val wordpieceTokens = encoder.encodeSentence(s, maxLength = maxSeqLength).take(maxSeqLength)
      WordpieceTokenizedSentence(wordpieceTokens)
    }
    sentenceTokenPieces
  }

}
