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

import scala.collection.JavaConverters._

/** XlnetEmbeddings (XLNet): Generalized Autoregressive Pretraining for Language Understanding
  *
  * Note that this is a very computationally expensive module compared to word embedding modules
  * that only perform embedding lookups. The use of an accelerator is recommended.
  *
  * XLNet is a new unsupervised language representation learning method based on a novel
  * generalized permutation language modeling objective. Additionally, XLNet employs
  * Transformer-XL as the backbone model, exhibiting excellent performance for language tasks
  * involving long context. Overall, XLNet achieves state-of-the-art (SOTA) results on various
  * downstream language tasks including question answering, natural language inference, sentiment
  * analysis, and document ranking.
  *
  * XLNet-Large =
  * [[https://storage.googleapis.com/xlnet/released_models/cased_L-24_H-1024_A-16.zip]] |
  * 24-layer, 1024-hidden, 16-heads XLNet-Base =
  * [[https://storage.googleapis.com/xlnet/released_models/cased_L-12_H-768_A-12.zip]] | 12-layer,
  * 768-hidden, 12-heads. This model is trained on full data (different from the one in the
  * paper).
  *
  * '''Sources :'''
  *
  * [[https://arxiv.org/abs/1906.08237]]
  *
  * [[https://github.com/zihangdai/xlnet]]
  *
  * '''Paper abstract: '''
  *
  * With the capability of modeling bidirectional contexts, denoising autoencoding based
  * pretraining like BERT achieves better performance than pretraining approaches based on
  * autoregressive language modeling. However, relying on corrupting the input with masks, BERT
  * neglects dependency between the masked positions and suffers from a pretrain-finetune
  * discrepancy. In light of these pros and cons, we propose XLNet, a generalized autoregressive
  * pretraining method that (1) enables learning bidirectional contexts by maximizing the expected
  * likelihood over all permutations of the factorization order and (2) overcomes the limitations
  * of BERT thanks to its autoregressive formulation. Furthermore, XLNet integrates ideas from
  * Transformer-XL, the state-of-the-art autoregressive model, into pretraining. Empirically,
  * under comparable experiment settings, XLNet outperforms BERT on 20 tasks, often by a large
  * margin, including question answering, natural language inference, sentiment analysis, and
  * document ranking. A list of (hyper-)parameter keys this annotator can take. Users can set and
  * get the parameter values through setters and getters, respectively.
  *
  * @param tensorflowWrapper
  *   XlmRoberta Model wrapper with TensorFlowWrapper
  * @param spp
  *   XlmRoberta SentencePiece model with SentencePieceWrapper
  * @param configProtoBytes
  *   Configuration for TensorFlow session
  * @param signatures
  *   Model's inputs and output(s) signatures
  */
private[johnsnowlabs] class Xlnet(
    val tensorflowWrapper: TensorflowWrapper,
    val spp: SentencePieceWrapper,
    configProtoBytes: Option[Array[Byte]] = None,
    signatures: Option[Map[String, String]] = None)
    extends Serializable {

  val _tfXlnetSignatures: Map[String, String] =
    signatures.getOrElse(ModelSignatureManager.apply())

  // keys representing the input and output tensors of the XLNet model
  private val SentenceStartTokenId = spp.getSppModel.pieceToId("<cls>")
  private val SentenceEndTokenId = spp.getSppModel.pieceToId("<sep>")
  private val SentencePadTokenId = spp.getSppModel.pieceToId("<pad>")
  private val SentencePieceDelimiterId = spp.getSppModel.pieceToId("▁")

  private def sessionWarmup(): Unit = {
    val dummyInput =
      Array(2834, 26, 23, 2458, 499, 18, 14976, 28, 18, 89, 25, 11574, 9, 4, 3)
    tag(Seq(dummyInput))
  }

  sessionWarmup()

  def tag(batch: Seq[Array[Int]]): Seq[Array[Array[Float]]] = {

    val maxSentenceLength = batch.map(pieceIds => pieceIds.length).max
    val batchLength = batch.length

    val tensors = new TensorResources()

    val (tokenTensors, maskTensors, segmentTensors) =
      PrepareEmbeddings.prepareBatchTensorsWithSegment(
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
        _tfXlnetSignatures.getOrElse(
          ModelSignatureConstants.InputIdsV1.key,
          "missing_input_id_key"),
        tokenTensors)
      .feed(
        _tfXlnetSignatures
          .getOrElse(ModelSignatureConstants.AttentionMaskV1.key, "missing_input_mask_key"),
        maskTensors)
      .feed(
        _tfXlnetSignatures
          .getOrElse(ModelSignatureConstants.TokenTypeIdsV1.key, "missing_segment_ids_key"),
        segmentTensors)
      .fetch(_tfXlnetSignatures
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
