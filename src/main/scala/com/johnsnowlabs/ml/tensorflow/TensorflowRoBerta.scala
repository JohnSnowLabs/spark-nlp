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

import com.johnsnowlabs.ml.tensorflow.sign.{ModelSignatureConstants, ModelSignatureManager}
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import org.tensorflow.ndarray.buffer.IntDataBuffer

import scala.collection.JavaConverters._

/**
 * The RoBERTa model was proposed in '''RoBERTa: A Robustly Optimized BERT Pretraining Approach''' [[https://arxiv.org/abs/1907.11692>]]
 * by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov.
 * It is based on Google's BERT model released in 2018.
 *
 * It builds on BERT and modifies key hyperparameters, removing the next-sentence pretraining objective and training with much larger mini-batches and learning rates.
 * The abstract from the paper is the following:
 *
 * Language model pretraining has led to significant performance gains but careful comparison between different
 * approaches is challenging. Training is computationally expensive, often done on private datasets of different sizes,
 * and, as we will show, hyperparameter choices have significant impact on the final results. We present a replication
 * study of BERT pretraining (Devlin et al., 2019) that carefully measures the impact of many key hyperparameters and
 * training data size. We find that BERT was significantly undertrained, and can match or exceed the performance of every
 * model published after it. Our best model achieves state-of-the-art results on GLUE, RACE and SQuAD. These results
 * highlight the importance of previously overlooked design choices, and raise questions about the source of recently
 * reported improvements. We release our models and code.*
 *
 * Tips:
 *
 * - RoBERTa has the same architecture as BERT, but uses a byte-level BPE as a tokenizer (same as GPT-2) and uses a different pretraining scheme.
 *
 * - RoBERTa doesn't have :obj:`token_type_ids`, you don't need to indicate which token belongs to which segment. Just separate your segments with the separation token :obj:`tokenizer.sep_token` (or :obj:`</s>`)
 *
 * The original code can be found ```here``` [[https://github.com/pytorch/fairseq/tree/master/examples/roberta]].
 *
 * @param tensorflowWrapper tensorflowWrapper class
 * @param sentenceStartTokenId special token id for `<s>`
 * @param sentenceEndTokenId special token id for `</s>`
 * @param configProtoBytes ProtoBytes for TensorFlow session config
 * @param signatures Model's inputs and output(s) signatures
 */
class TensorflowRoBerta(val tensorflowWrapper: TensorflowWrapper,
                        sentenceStartTokenId: Int,
                        sentenceEndTokenId: Int,
                        padTokenId: Int,
                        configProtoBytes: Option[Array[Byte]] = None,
                        signatures: Option[Map[String, String]] = None
                       ) extends Serializable {

  val _tfRoBertaSignatures: Map[String, String] = signatures.getOrElse(ModelSignatureManager.apply())

  /** Encode the input sequence to indexes IDs adding padding where necessary */
  def encode(sentences: Seq[(WordpieceTokenizedSentence, Int)], maxSequenceLength: Int): Seq[Array[Int]] = {
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
        maskBuffers.offset(offset).write(sentence.map(x => if (x == padTokenId) 0 else 1))
      }

    val runner = tensorflowWrapper.getTFHubSession(configProtoBytes = configProtoBytes, initAllTables = false).runner

    val tokenTensors = tensors.createIntBufferTensor(shape, tokenBuffers)
    val maskTensors = tensors.createIntBufferTensor(shape, maskBuffers)

    runner
      .feed(_tfRoBertaSignatures.getOrElse(ModelSignatureConstants.InputIds.key, "missing_input_id_key"), tokenTensors)
      .feed(_tfRoBertaSignatures.getOrElse(ModelSignatureConstants.AttentionMask.key, "missing_input_mask_key"), maskTensors)
      .fetch(_tfRoBertaSignatures.getOrElse(ModelSignatureConstants.LastHiddenState.key, "missing_sequence_output_key"))

    val outs = runner.run().asScala
    val embeddings = TensorResources.extractFloats(outs.head)

    outs.foreach(_.close())
    tensors.clearSession(outs)
    tensors.clearTensors()

    val dim = embeddings.length / (batchLength * maxSentenceLength)
    val shrinkedEmbeddings: Array[Array[Array[Float]]] = embeddings.grouped(dim).toArray.grouped(maxSentenceLength).toArray

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

  /**
   *
   * @param batch batches of sentences
   * @return batches of vectors for each sentence
   */
  def tagSentence(batch: Seq[Array[Int]]): Array[Array[Float]] = {
    val tensors = new TensorResources()
    val tensorsMasks = new TensorResources()

    val maxSentenceLength = batch.map(x => x.length).max
    val batchLength = batch.length

    val tokenBuffers = tensors.createIntBuffer(batchLength * maxSentenceLength)
    val maskBuffers = tensorsMasks.createIntBuffer(batchLength * maxSentenceLength)


    val shape = Array(batchLength.toLong, maxSentenceLength)

    batch.zipWithIndex.foreach { case (sentence, idx) =>
      val offset = idx * maxSentenceLength

      tokenBuffers.offset(offset).write(sentence)
      maskBuffers.offset(offset).write(sentence.map(x => if (x == 0) 0 else 1))
    }

    val runner = tensorflowWrapper.getTFHubSession(configProtoBytes = configProtoBytes, initAllTables = false).runner

    val tokenTensors = tensors.createIntBufferTensor(shape, tokenBuffers)
    val maskTensors = tensorsMasks.createIntBufferTensor(shape, maskBuffers)

    runner
      .feed(_tfRoBertaSignatures.getOrElse(ModelSignatureConstants.InputIds.key, "missing_input_id_key"), tokenTensors)
      .feed(_tfRoBertaSignatures.getOrElse(ModelSignatureConstants.AttentionMask.key, "missing_input_mask_key"), maskTensors)
      .fetch(_tfRoBertaSignatures.getOrElse(ModelSignatureConstants.PoolerOutput.key, "missing_pooled_output_key"))

    val outs = runner.run().asScala
    val embeddings = TensorResources.extractFloats(outs.head)

    tensors.clearSession(outs)
    tensors.clearTensors()

    val dim = embeddings.length / batchLength
    embeddings.grouped(dim).toArray

  }

  def calculateEmbeddings(sentences: Seq[WordpieceTokenizedSentence],
                          originalTokenSentences: Seq[TokenizedSentence],
                          batchSize: Int,
                          maxSentenceLength: Int,
                          caseSensitive: Boolean
                         ): Seq[WordpieceEmbeddingsSentence] = {

    /*Run embeddings calculation by batches*/
    sentences.zipWithIndex.grouped(batchSize).flatMap { batch =>
      val encoded = encode(batch, maxSentenceLength)
      val vectors = tag(encoded)

      /*Combine tokens and calculated embeddings*/
      batch.zip(vectors).map { case (sentence, tokenVectors) =>
        val tokenLength = sentence._1.tokens.length

        /*All wordpiece embeddings*/
        val tokenEmbeddings = tokenVectors.slice(1, tokenLength + 1)

        /*Word-level and span-level alignment with Tokenizer
        https://github.com/google-research/bert#tokenization

        ### Input
        orig_tokens = ["John", "Johanson", "'s",  "house"]
        labels      = ["NNP",  "NNP",      "POS", "NN"]

        # bert_tokens == ["[CLS]", "john", "johan", "##son", "'", "s", "house", "[SEP]"]
        # orig_to_tok_map == [1, 2, 4, 6]*/

        val tokensWithEmbeddings = sentence._1.tokens.zip(tokenEmbeddings).flatMap {
          case (token, tokenEmbedding) =>
            val tokenWithEmbeddings = TokenPieceEmbeddings(token, tokenEmbedding)
            val originalTokensWithEmbeddings = originalTokenSentences(sentence._2).indexedTokens.find(
              p => p.begin == tokenWithEmbeddings.begin).map {
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

  def calculateSentenceEmbeddings(tokens: Seq[WordpieceTokenizedSentence],
                                  sentences: Seq[Sentence],
                                  batchSize: Int,
                                  maxSentenceLength: Int
                                 ): Seq[Annotation] = {

    /*Run embeddings calculation by batches*/
    tokens.zip(sentences).zipWithIndex.grouped(batchSize).flatMap { batch =>
      val tokensBatch = batch.map(x => (x._1._1, x._2))
      val sentencesBatch = batch.map(x => x._1._2)
      val encoded = encode(tokensBatch, maxSentenceLength)
      val embeddings = tagSentence(encoded)

      sentencesBatch.zip(embeddings).map { case (sentence, vectors) =>
        Annotation(
          annotatorType = AnnotatorType.SENTENCE_EMBEDDINGS,
          begin = sentence.start,
          end = sentence.end,
          result = sentence.content,
          metadata = Map("sentence" -> sentence.index.toString,
            "token" -> sentence.content,
            "pieceId" -> "-1",
            "isWordStart" -> "true"
          ),
          embeddings = vectors
        )
      }
    }.toSeq
  }

}


