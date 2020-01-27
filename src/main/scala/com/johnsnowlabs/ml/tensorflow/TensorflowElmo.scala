package com.johnsnowlabs.ml.tensorflow

import com.johnsnowlabs.nlp.annotators.common._

import scala.collection.JavaConverters._

/**
  * This class is used to calculate ELMO embeddings for For Sequence Batches of TokenizedSentences.
  *
  * https://tfhub.dev/google/elmo/3
  * * word_emb: the character-based word representations with shape [batch_size, max_length, 512].  == -3
  * * lstm_outputs1: the first LSTM hidden state with shape [batch_size, max_length, 1024]. === -2
  * * lstm_outputs2: the second LSTM hidden state with shape [batch_size, max_length, 1024]. === -1
  * * elmo: the weighted sum of the 3 layers, where the weights are trainable. This tensor has shape [batch_size, max_length, 1024]  == 0
  *
  * @param tensorflow           Elmo Model wrapper with Tensorflow Wrapper
  * @param sentenceStartTokenId Start Token ID
  * @param sentenceEndTokenId   End Token ID
  * @param maxSentenceLength    Max sentence length
  * @param batchSize            size of batch
  * @param configProtoBytes     Configuration for Tensorflow session
  */

class TensorflowElmo(val tensorflow: TensorflowWrapper,
                     sentenceStartTokenId: Int,
                     sentenceEndTokenId: Int,
                     maxSentenceLength: Int,
                     batchSize: Int,
                     configProtoBytes: Option[Array[Byte]] = None
                    ) extends Serializable {

  private val tokenIdsKey = "input"


  /**
    * Calculate the embeddigns for a sequence of Tokens and create WordPieceEmbeddingsSentence objects from them
    *
    * @param sentences    A sequence of Tokenized Sentences for which embeddings will be calculated
    * @param poolingLayer Define which output layer you want from the model -3 = word_emb, -2 = lstm1, -1=lmst2 , 0=elmo. See https://tfhub.dev/google/elmo/3 for reference
    * @return A Seq of WordpieceEmbeddingsSentence, one element for each input sentence
    */
  def calculateEmbeddings(sentences: Seq[TokenizedSentence], poolingLayer: Int): Seq[WordpieceEmbeddingsSentence] = {

    /*Run embeddings calculation by batches*/
    sentences.zipWithIndex.grouped(batchSize).flatMap { batch =>
      val vectors = tag((sentences), extractPoolingLayer(poolingLayer))
      /*Combine tokens and sentences  and their calculated embeddings*/
      batch.zip(vectors).map { case (sentence, tokenVectors) =>
        val tokenLength = sentence._1.indexedTokens.length


        /*All wordpiece embeddings and sentence embeddings*/
        val tokenEmbeddings = tokenVectors.slice(0, tokenLength + 1)
        val tokensWithEmbeddings = sentence._1.indexedTokens.zip(tokenEmbeddings).map {
          case (token, tokenEmbedding) =>
            TokenPieceEmbeddings(
              token.token,
              token.token,
              -1,
              isWordStart = true,
              false,
              tokenEmbedding,
              token.begin,
              token.end
            )

        }
        WordpieceEmbeddingsSentence(tokensWithEmbeddings, sentence._1.sentenceIndex)
      }
    }.toSeq

  }

  /**
    * Tag a seq of TokenizedSentences, will get the embeddings according to key.
    *
    * @param batch         The Tokens for which we calculate embeddings
    * @param embeddingsKey Specification of the output embedding for Elmo
    * @return The Embeddings Vector. For each Seq Element we have a Sentence, and for each sentence we have an Array for each of its words. Each of its words gets a float array to represent its Embeddings
    */
  def tag(batch: Seq[TokenizedSentence], embeddingsKey: String): Seq[Array[Array[Float]]] = {

    val tensors = new TensorResources()
    val sentencesBytes = batch.flatMap { sentence =>
      sentence.indexedTokens.map { token =>
        token.token.getBytes("UTF-8")
      }
    }.toArray

    val sentenceTensors = tensors.createTensor(sentencesBytes)

    val runner = tensorflow.getSession(configProtoBytes = configProtoBytes).runner

    runner
      .feed(tokenIdsKey, sentenceTensors)
      .fetch(embeddingsKey)

    val outs = runner.run().asScala
    val wordEmbeddings = TensorResources.extractFloats(outs.head)
    tensors.clearTensors()

    val wordDim = 512 // batchSize TODO add batchsize?

    val shrinkedWordEmbeddings: Array[Array[Array[Float]]] = wordEmbeddings.grouped(wordDim).toArray.grouped(maxSentenceLength).toArray

    val emptyVector = Array.fill(1024 + 512)(0f)

    batch.zip(shrinkedWordEmbeddings).map { case (ids, embeddings) =>
      if (ids.indexedTokens.length > embeddings.length) {
        embeddings.take(embeddings.length - 1) ++
          Array.fill(embeddings.length - ids.indexedTokens.length)(emptyVector) ++
          Array(embeddings.last)
      } else {
        embeddings
      }
    }

  }

  /**
    * word_emb: the character-based word representations with shape [batch_size, max_length, 512].  == -3
    * lstm_outputs1: the first LSTM hidden state with shape [batch_size, max_length, 1024]. === -2
    * lstm_outputs2: the second LSTM hidden state with shape [batch_size, max_length, 1024]. === -1
    * elmo: the weighted sum of the 3 layers, where the weights are trainable. This tensor has shape [batch_size, max_length, 1024]  == 0
    *
    * @param layer Layer specification
    * @return The key to use to get the embeddign
    */
  def extractPoolingLayer(layer: Int): String = {
    {
      layer match {
        case -3 =>
          "word_emb"
        case -2 =>
          "lstm1"
        case -1 =>
          "lstm12"
        case 0 =>
          "elmo"

      }
    }

  }
}