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

package com.johnsnowlabs.ml.tensorflow

import com.johnsnowlabs.nlp.annotators.common._

import scala.collection.JavaConverters._

/** Embeddings from a language model trained on the 1 Billion Word Benchmark.
  *
  * Note that this is a very computationally expensive module compared to word embedding modules
  * that only perform embedding lookups. The use of an accelerator is recommended.
  *
  * '''word_emb''': the character-based word representations with shape [batch_size, max_length,
  * 512]. == word_emb
  *
  * '''lstm_outputs1''': the first LSTM hidden state with shape [batch_size, max_length, 1024].
  * \=== lstm_outputs1
  *
  * '''lstm_outputs2''': the second LSTM hidden state with shape [batch_size, max_length, 1024].
  * \=== lstm_outputs2
  *
  * '''elmo''': the weighted sum of the 3 layers, where the weights are trainable. This tensor has
  * shape [batch_size, max_length, 1024] == elmo
  *
  * See
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/ElmoEmbeddingsTestSpec.scala]]
  * for further reference on how to use this API.
  *
  * @param tensorflow
  *   Elmo Model wrapper with TensorFlow Wrapper
  * @param batchSize
  *   size of batch
  * @param configProtoBytes
  *   Configuration for TensorFlow session
  *
  * Sources :
  *
  * [[https://tfhub.dev/google/elmo/3]]
  *
  * [[https://arxiv.org/abs/1802.05365]]
  */
class TensorflowElmo(
    val tensorflow: TensorflowWrapper,
    batchSize: Int,
    configProtoBytes: Option[Array[Byte]] = None)
    extends Serializable {

  private val TokensKey = "tokens"
  private val SequenceKey = "sequence_len"

  /** Calculate the embeddigns for a sequence of Tokens and create WordPieceEmbeddingsSentence
    * objects from them
    *
    * @param sentences
    *   A sequence of Tokenized Sentences for which embeddings will be calculated
    * @param poolingLayer
    *   Define which output layer you want from the model word_emb, lstm_outputs1, lstm_outputs2,
    *   elmo. See https://tfhub.dev/google/elmo/3 for reference
    * @return
    *   A Seq of WordpieceEmbeddingsSentence, one element for each input sentence
    */
  def predict(
      sentences: Seq[TokenizedSentence],
      poolingLayer: String): Seq[WordpieceEmbeddingsSentence] = {

    /*Run embeddings calculation by batches*/
    sentences.zipWithIndex
      .grouped(batchSize)
      .flatMap { batch =>
        val vectors = tag((sentences), poolingLayer, getDimensions(poolingLayer))
        /*Combine tokens and sentences  and their calculated embeddings*/
        batch.zip(vectors).map { case (sentence, tokenVectors) =>
          val tokenLength = sentence._1.indexedTokens.length
          val tokenEmbeddings = tokenVectors.slice(0, tokenLength)

          val tokensWithEmbeddings = sentence._1.indexedTokens
            .zip(tokenEmbeddings)
            .map { case (token, tokenEmbeddings) =>
              TokenPieceEmbeddings(
                token.token,
                token.token,
                -1,
                isWordStart = true,
                isOOV = false,
                tokenEmbeddings,
                token.begin,
                token.end)
            }
          WordpieceEmbeddingsSentence(tokensWithEmbeddings, sentence._1.sentenceIndex)
        }
      }
      .toSeq

  }

  /** Tag a seq of TokenizedSentences, will get the embeddings according to key.
    *
    * @param batch
    *   The Tokens for which we calculate embeddings
    * @param embeddingsKey
    *   Specification of the output embedding for Elmo
    * @param dimension
    *   Elmo's embeddings dimension: either 512 or 1024
    * @return
    *   The Embeddings Vector. For each Seq Element we have a Sentence, and for each sentence we
    *   have an Array for each of its words. Each of its words gets a float array to represent its
    *   Embeddings
    */
  def tag(
      batch: Seq[TokenizedSentence],
      embeddingsKey: String,
      dimension: Int): Seq[Array[Array[Float]]] = {

    val tensors = new TensorResources()

    /* Actual size of each sentence to skip padding in the TF model */
    val sequencesLength = batch.map(x => x.indexedTokens.length).toArray
    val maxSentenceLength = sequencesLength.max

    val sentencesBytes = batch.map { sentence =>
      val indexedToks = sentence.indexedTokens
      val diff = maxSentenceLength - indexedToks.length

      if (indexedToks.length < maxSentenceLength) {
        val tokens = indexedToks.map {
          _.token
        }
        /* Padding by adding extra empty tokens to smaller sentences */
        tokens ++ Array.fill(1, diff)(" ").head
      } else {
        indexedToks.map {
          _.token
        }
      }
    }.toArray

    val sentenceTensors = tensors.createTensor(sentencesBytes)
    val runner = tensorflow.getTFSession(configProtoBytes = configProtoBytes).runner

    runner
      .feed(TokensKey, sentenceTensors)
      .feed(SequenceKey, tensors.createTensor(sequencesLength))
      .fetch(embeddingsKey)

    val outs = runner.run().asScala
    val wordEmbeddings = TensorResources.extractFloats(outs.head)
    tensors.clearSession(outs)
    tensors.clearTensors()

    var allWordEmbeddings: Array[Array[Float]] = wordEmbeddings.grouped(dimension).toArray

    /* Group embeddings based on the length of the sentence */
    val embeddingsBySentence = batch.map { case sentence =>
      val embds = allWordEmbeddings.slice(0, sentence.indexedTokens.length)
      /* Remove the already used vectors */
      allWordEmbeddings = allWordEmbeddings.slice(0, sentence.indexedTokens.length * dimension)
      embds
    }

    batch.zip(embeddingsBySentence).map { case (_, embeddings) => embeddings }
  }

  /** word_emb: the character-based word representations with shape [batch_size, max_length, 512].
    * \== 512
    *
    * lstm_outputs1: the first LSTM hidden state with shape [batch_size, max_length, 1024]. ===
    * 1024
    *
    * lstm_outputs2: the second LSTM hidden state with shape [batch_size, max_length, 1024]. ===
    * 1024
    *
    * elmo: the weighted sum of the 3 layers, where the weights are trainable. This tensor has
    * shape [batch_size, max_length, 1024] == 1024
    *
    * @return
    *   The dimension of chosen layer
    */
  def getDimensions: String => Int = {
    case "word_emb" => 512
    case "lstm_outputs1" => 1024
    case "lstm_outputs2" => 1024
    case "elmo" => 1024
  }
}
