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
import com.johnsnowlabs.nlp.annotators.common._

import scala.collection.JavaConverters._

/** XlnetEmbeddings (XLNet): Generalized Autoregressive Pretraining for Language Understanding
 *
 * Note that this is a very computationally expensive module compared to word embedding modules that only perform embedding lookups.
 * The use of an accelerator is recommended.
 *
 * XLNet is a new unsupervised language representation learning method based on a novel generalized permutation language modeling objective. Additionally, XLNet employs Transformer-XL as the backbone model, exhibiting excellent performance for language tasks involving long context. Overall, XLNet achieves state-of-the-art (SOTA) results on various downstream language tasks including question answering, natural language inference, sentiment analysis, and document ranking.
 *
 * XLNet-Large     = [[https://storage.googleapis.com/xlnet/released_models/cased_L-24_H-1024_A-16.zip]]    | 24-layer, 1024-hidden, 16-heads
 * XLNet-Base    = [[https://storage.googleapis.com/xlnet/released_models/cased_L-12_H-768_A-12.zip]]   |  12-layer, 768-hidden, 12-heads. This model is trained on full data (different from the one in the paper).
 *
 * @param uid required internal uid for saving annotator
 *
 *            '''Sources :'''
 *
 *            [[ https://arxiv.org/abs/1906.08237]]
 *
 *            [[ https://github.com/zihangdai/xlnet]]
 *
 *            '''Paper abstract: '''
 *
 *            With the capability of modeling bidirectional contexts, denoising autoencoding based pretraining like BERT achieves better performance than pretraining approaches based on autoregressive language modeling. However, relying on corrupting the input with masks, BERT neglects dependency between the masked positions and suffers from a pretrain-finetune discrepancy. In light of these pros and cons, we propose XLNet, a generalized autoregressive pretraining method that (1) enables learning bidirectional contexts by maximizing the expected likelihood over all permutations of the factorization order and (2) overcomes the limitations of BERT thanks to its autoregressive formulation. Furthermore, XLNet integrates ideas from Transformer-XL, the state-of-the-art autoregressive model, into pretraining. Empirically, under comparable experiment settings, XLNet outperforms BERT on 20 tasks, often by a large margin, including question answering, natural language inference, sentiment analysis, and document ranking.
 * @groupname anno Annotator types
 * @groupdesc anno Required input and expected output annotator types
 * @groupname Ungrouped Members
 * @groupname param Parameters
 * @groupname setParam Parameter setters
 * @groupname getParam Parameter getters
 * @groupname Ungrouped Members
 * @groupprio param  1
 * @groupprio anno  2
 * @groupprio Ungrouped 3
 * @groupprio setParam  4
 * @groupprio getParam  5
 * @groupdesc param A list of (hyper-)parameter keys this annotator can take. Users can set and get the parameter values through setters and getters, respectively.
 */
class TensorflowXlnet(val tensorflow: TensorflowWrapper,
                      val spp: SentencePieceWrapper,
                      configProtoBytes: Option[Array[Byte]] = None
                     ) extends Serializable {

  // keys representing the input and output tensors of the XLNet model
  private val tokenIdsKey = "input_ids"
  private val maskIdsKey = "input_mask"
  private val segmentIdsKey = "segment_ids"
  private val outputSequenceKey = "module/seq_out"

  private val tokenSEPCLSIds = Array(4, 3)
  private val sentencePieceDelimiterId = 17

  def getSpecialTokens(token: String): Array[Int] = {
    spp.getSppModel.encodeAsIds(token)
  }

  def prepareBatchInputs(sentences: Seq[(WordpieceTokenizedSentence, Int)], maxSequenceLength: Int): Seq[Array[Int]] = {
    val maxSentenceLength =
      Array(
        maxSequenceLength - 2,
        sentences.map{ case(wpTokSentence, _) => wpTokSentence.tokens.length}.max).min

    sentences
      .map{ case(wpTokSentence, _) =>
        val tokenPieceIds = wpTokSentence.tokens.map(t => t.pieceId)
        val padding = Array.fill(maxSentenceLength - tokenPieceIds.length)(0)

        tokenPieceIds.take(maxSentenceLength) ++ tokenSEPCLSIds ++ padding
      }
  }

  def tag(batch: Seq[Array[Int]]): Seq[Array[Array[Float]]] = {

    val tensors = new TensorResources()

    /* Actual size of each sentence to skip padding in the TF model */
    val sequencesLength = batch.map(x => x.length).toArray
    val maxSentenceLength = sequencesLength.max

    val tokenBuffers = tensors.createIntBuffer(batch.length*maxSentenceLength)
    val maskBuffers = tensors.createFloatBuffer(batch.length*maxSentenceLength)
    val segmentBuffers = tensors.createIntBuffer(batch.length*maxSentenceLength)

    val shape = Array(batch.length.toLong, maxSentenceLength)

    batch.zipWithIndex.foreach { case(tokenIds, idx) =>
      val offset = idx * maxSentenceLength
      val diff = maxSentenceLength - tokenIds.length
      segmentBuffers.offset(offset).write(Array.fill(maxSentenceLength)(0))

      val padding = Array.fill(diff)(0)
      val newTokenIds = tokenIds ++ padding

      tokenBuffers.offset(offset).write(newTokenIds)
      maskBuffers.offset(offset).write(newTokenIds.map(x=> if (x == 0) 0f else 1f))
    }


    val tokenTensors = tensors.createIntBufferTensor(shape, tokenBuffers)
    val maskTensors = tensors.createFloatBufferTensor(shape, maskBuffers)
    val segmentTensors = tensors.createIntBufferTensor(shape, segmentBuffers)

    val runner = tensorflow.getTFHubSession(configProtoBytes = configProtoBytes).runner

    runner
      .feed(tokenIdsKey, tokenTensors)
      .feed(maskIdsKey, maskTensors)
      .feed(segmentIdsKey, segmentTensors)
      .fetch(outputSequenceKey)

    val outs = runner.run().asScala
    val embeddings = TensorResources.extractFloats(outs.head)

    tensors.clearSession(outs)
    tensors.clearTensors()
    tokenTensors.close()
    maskTensors.close()
    segmentTensors.close()

    val dim = embeddings.length / (batch.length * maxSentenceLength)
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
      batch.zip(vectors).map{case (sentence, tokenVectors) =>
        val tokenLength = sentence._1.tokens.length
        /*All wordpiece embeddings*/
        val tokenEmbeddings = tokenVectors.slice(1, tokenLength + 1)
        val tokensWithEmbeddings = sentence._1.tokens.zip(tokenEmbeddings).flatMap{
          case (token, tokenEmbedding) =>
            val tokenWithEmbeddings = TokenPieceEmbeddings(token, tokenEmbedding)
            val originalTokensWithEmbeddings = tokenizedSentences(sentence._2).indexedTokens.find(
              p => p.begin == tokenWithEmbeddings.begin).map{
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
    val encoder = new SentencepieceEncoder(spp, caseSensitive, sentencePieceDelimiterId)

    val sentecneTokenPieces = sentences.map { s =>
      val shrinkedSentence = s.indexedTokens.take(maxSeqLength - 2)
      val wordpieceTokens = shrinkedSentence.flatMap(token => encoder.encode(token)).take(maxSeqLength)
      WordpieceTokenizedSentence(wordpieceTokens)
    }
    sentecneTokenPieces
  }

}
