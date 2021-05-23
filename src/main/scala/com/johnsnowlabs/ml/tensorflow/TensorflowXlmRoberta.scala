/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
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

import com.johnsnowlabs.ml.tensorflow.sentencepiece._
import com.johnsnowlabs.ml.tensorflow.sign.{ModelSignatureConstants, ModelSignatureManager}
import com.johnsnowlabs.nlp.annotators.common._
import org.tensorflow.ndarray.buffer.DataBuffers

import scala.collection.JavaConverters._

/**
 * The XLM-RoBERTa model was proposed in '''Unsupervised Cross-lingual Representation Learning at Scale'''
 * [[https://arxiv.org/abs/1911.02116]] by Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume
 * Wenzek, Francisco GuzmÃ¡n, Edouard Grave, Myle Ott, Luke Zettlemoyer and Veselin Stoyanov. It is based on Facebook's
 * RoBERTa model released in 2019. It is a large multi-lingual language model, trained on 2.5TB of filtered CommonCrawl
 * data.
 *
 * The abstract from the paper is the following:
 *
 * This paper shows that pretraining multilingual language models at scale leads to significant performance gains for a
 * wide range of cross-lingual transfer tasks. We train a Transformer-based masked language model on one hundred
 * languages, using more than two terabytes of filtered CommonCrawl data. Our model, dubbed XLM-R, significantly
 * outperforms multilingual BERT (mBERT) on a variety of cross-lingual benchmarks, including +13.8% average accuracy on
 * XNLI, +12.3% average F1 score on MLQA, and +2.1% average F1 score on NER. XLM-R performs particularly well on
 * low-resource languages, improving 11.8% in XNLI accuracy for Swahili and 9.2% for Urdu over the previous XLM model. We
 * also present a detailed empirical evaluation of the key factors that are required to achieve these gains, including the
 * trade-offs between (1) positive transfer and capacity dilution and (2) the performance of high and low resource
 * languages at scale. Finally, we show, for the first time, the possibility of multilingual modeling without sacrificing
 * per-language performance; XLM-Ris very competitive with strong monolingual models on the GLUE and XNLI benchmarks. We
 * will make XLM-R code, data, and models publicly available.
 *
 * Tips:
 *
 * - XLM-RoBERTa is a multilingual model trained on 100 different languages. Unlike some XLM multilingual models, it does
 * not require '''lang''' parameter to understand which language is used, and should be able to determine the correct
 * language from the input ids.
 * - This implementation is the same as RoBERTa. Refer to the [[com.johnsnowlabs.nlp.embeddings.RoBertaEmbeddings]] for usage examples
 * as well as the information relative to the inputs and outputs.
 *
 * @param tensorflowWrapper    XlmRoberta Model wrapper with TensorFlowWrapper
 * @param spp                  XlmRoberta SentencePiece model with SentencePieceWrapper
 * @param batchSize            size of batch
 * @param sentenceStartTokenId piece id for starting sequence '''<s>'''
 * @param sentenceEndTokenId   piece id for ending sequence '''</s>'''
 * @param padTokenId           piece id for padding '''<pad>'''
 * @param configProtoBytes     Configuration for TensorFlow session
 */

class TensorflowXlmRoberta(val tensorflowWrapper: TensorflowWrapper,
                           val spp: SentencePieceWrapper,
                           batchSize: Int,
                           sentenceStartTokenId: Int,
                           sentenceEndTokenId: Int,
                           padTokenId: Int,
                           configProtoBytes: Option[Array[Byte]] = None,
                           signatures: Option[Map[String, String]] = None
                          ) extends Serializable {

  val _tfRoBertaSignatures: Map[String, String] = signatures.getOrElse(ModelSignatureManager.apply())

  private val SentencePieceDelimiterId = 13

  def prepareBatchInputs(sentences: Seq[(WordpieceTokenizedSentence, Int)], maxSequenceLength: Int): Seq[Array[Int]] = {
    val maxSentenceLength =
      Array(
        maxSequenceLength - 2,
        sentences.map { case (wpTokSentence, _) => wpTokSentence.tokens.length }.max).min

    sentences
      .map { case (wpTokSentence, _) =>
        val tokenPieceIds = wpTokSentence.tokens.map(t => t.pieceId)
        val padding = Array.fill(maxSentenceLength - tokenPieceIds.length)(padTokenId)

        Array(sentenceStartTokenId) ++ tokenPieceIds.take(maxSentenceLength) ++ Array(sentenceEndTokenId) ++ padding
      }
  }

  def tag(batch: Seq[Array[Int]]): Seq[Array[Array[Float]]] = {

    val tensors = new TensorResources()
    val tensorsMasks = new TensorResources()

    /* Actual size of each sentence to skip padding in the TF model */
    val sequencesLength = batch.map(x => x.length).toArray
    val maxSentenceLength = sequencesLength.max

    val tokenBuffers = DataBuffers.ofInts(batch.length * maxSentenceLength)
    val maskBuffers = DataBuffers.ofInts(batch.length * maxSentenceLength)

    val shape = Array(batch.length.toLong, maxSentenceLength)

    batch.zipWithIndex
      .foreach { case (tokenIds, idx) =>
        val offset = idx * maxSentenceLength
        val diff = maxSentenceLength - tokenIds.length

        val padding = Array.fill(diff)(padTokenId)
        val newTokenIds = tokenIds ++ padding

        tokenBuffers.offset(offset).write(newTokenIds)
        maskBuffers.offset(offset).write(newTokenIds.map(x => if (x == padTokenId) 0 else 1))
      }

    val tokenTensors = tensors.createIntBufferTensor(shape, tokenBuffers)
    val maskTensors = tensorsMasks.createIntBufferTensor(shape, maskBuffers)

    val runner = tensorflowWrapper.getTFHubSession(configProtoBytes = configProtoBytes).runner

    runner
      .feed(_tfRoBertaSignatures.getOrElse(ModelSignatureConstants.InputIds.key, "missing_input_id_key"), tokenTensors)
      .feed(_tfRoBertaSignatures.getOrElse(ModelSignatureConstants.AttentionMask.key, "missing_input_mask_key"), maskTensors)
      .fetch(_tfRoBertaSignatures.getOrElse(ModelSignatureConstants.LastHiddenState.key, "missing_sequence_output_key"))

    val outs = runner.run().asScala
    val embeddings = TensorResources.extractFloats(outs.head)

    tensors.clearSession(outs)
    tensors.clearTensors()

    val dim = embeddings.length / (batch.length * maxSentenceLength)
    val shrinkedEmbeddings: Array[Array[Array[Float]]] =
      embeddings
        .grouped(dim).toArray
        .grouped(maxSentenceLength).toArray

    val emptyVector = Array.fill(dim)(0f)

    batch.zip(shrinkedEmbeddings).map { case (ids, embeddings) =>
      if (ids.length > embeddings.length) {
        embeddings.take(embeddings.length - 1) ++
          Array.fill(embeddings.length - ids.length)(emptyVector) ++
          Array(embeddings.last)
      } else {
        embeddings
      }
    }
  }

  def calculateEmbeddings(tokenizedSentences: Seq[TokenizedSentence],
                          batchSize: Int,
                          maxSentenceLength: Int,
                          caseSensitive: Boolean
                         ): Seq[WordpieceEmbeddingsSentence] = {

    val wordPieceTokenizedSentences = tokenizeWithAlignment(tokenizedSentences, maxSentenceLength, caseSensitive)
    wordPieceTokenizedSentences.zipWithIndex.grouped(batchSize).flatMap { batch =>
      val batchedInputsIds = prepareBatchInputs(batch, maxSentenceLength)
      val vectors = tag(batchedInputsIds)

      /*Combine tokens and calculated embeddings*/
      batch.zip(vectors).map { case (sentence, tokenVectors) =>
        val tokenLength = sentence._1.tokens.length
        /*All wordpiece embeddings*/
        val tokenEmbeddings = tokenVectors.slice(1, tokenLength + 1)
        val tokensWithEmbeddings = sentence._1.tokens.zip(tokenEmbeddings).flatMap {
          case (token, tokenEmbedding) =>
            val tokenWithEmbeddings = TokenPieceEmbeddings(token, tokenEmbedding)
            val originalTokensWithEmbeddings = tokenizedSentences(sentence._2).indexedTokens.find(
              p => p.begin == tokenWithEmbeddings.begin && tokenWithEmbeddings.isWordStart).map {
              token =>
                val originalTokenWithEmbedding = TokenPieceEmbeddings(
                  TokenPiece(wordpiece = tokenWithEmbeddings.wordpiece,
                    token = if (caseSensitive) token.token else token.token.toLowerCase(),
                    pieceId = tokenWithEmbeddings.pieceId,
                    isWordStart = tokenWithEmbeddings.isWordStart,
                    begin = token.begin,
                    end = token.end
                  ),
                  tokenEmbedding
                )
                originalTokenWithEmbedding
            }
            originalTokensWithEmbeddings
        }

        WordpieceEmbeddingsSentence(tokensWithEmbeddings, sentence._2)
      }
    }.toSeq
  }

  def tokenizeWithAlignment(sentences: Seq[TokenizedSentence], maxSeqLength: Int, caseSensitive: Boolean): Seq[WordpieceTokenizedSentence] = {
    val encoder = new SentencepieceEncoder(spp, caseSensitive, 6, pieceIdFromZero = true)

    val sentecneTokenPieces = sentences.map { s =>
      val shrinkedSentence = s.indexedTokens.take(maxSeqLength - 2)
      val wordpieceTokens = shrinkedSentence.flatMap(token => encoder.encode(token)).take(maxSeqLength)
      WordpieceTokenizedSentence(wordpieceTokens)
    }
    sentecneTokenPieces
  }

}
