package com.johnsnowlabs.ml.tensorflow

import com.johnsnowlabs.nlp.annotators.common._

import scala.collection.JavaConverters._

/**
  * This class is used to calculate ELMO embeddings for For Sequence Batches of TokenizedSentences.
  *
  * https://tfhub.dev/google/elmo/3
  * * word_emb: the character-based word representations with shape [batch_size, max_length, 512].  == word_emb
  * * lstm_outputs1: the first LSTM hidden state with shape [batch_size, max_length, 1024]. === lstm_outputs1
  * * lstm_outputs2: the second LSTM hidden state with shape [batch_size, max_length, 1024]. === lstm_outputs2
  * * elmo: the weighted sum of the 3 layers, where the weights are trainable. This tensor has shape [batch_size, max_length, 1024]  == elmo
  *
  * @param tensorflow           Elmo Model wrapper with TensorFlow Wrapper
  * @param batchSize            size of batch
  * @param configProtoBytes     Configuration for TensorFlow session
  */

class TensorflowElmo(val tensorflow: TensorflowWrapper,
                     batchSize: Int,
                     configProtoBytes: Option[Array[Byte]] = None
                    ) extends Serializable {

  private val tokensKey = "tokens"
  private val sequenceKey = "sequence_len"


  /**
    * Calculate the embeddigns for a sequence of Tokens and create WordPieceEmbeddingsSentence objects from them
    *
    * @param sentences    A sequence of Tokenized Sentences for which embeddings will be calculated
    * @param poolingLayer Define which output layer you want from the model word_emb, lstm_outputs1, lstm_outputs2, elmo. See https://tfhub.dev/google/elmo/3 for reference
    * @return A Seq of WordpieceEmbeddingsSentence, one element for each input sentence
    */
  def calculateEmbeddings(sentences: Seq[TokenizedSentence], poolingLayer: String): Seq[WordpieceEmbeddingsSentence] = {

    /*Run embeddings calculation by batches*/
    sentences.zipWithIndex.grouped(batchSize).flatMap { batch =>
      val vectors = tag((sentences), poolingLayer, getDimensions(poolingLayer))
      /*Combine tokens and sentences  and their calculated embeddings*/
      batch.zip(vectors).map { case (sentence, tokenVectors) =>
        val tokenLength = sentence._1.indexedTokens.length
        //        val endRange =
        val tokenEmbeddings = tokenVectors.slice(0, tokenLength)

        val tokensWithEmbeddings = sentence._1.indexedTokens.zip(tokenEmbeddings).map {
          case (token, tokenEmbedding) =>
            TokenPieceEmbeddings(
              token.token,
              token.token,
              -1,
              isWordStart = true,
              isOOV = false,
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
    * @param dimension Elmo's embeddings dimension: either 512 or 1024
    * @return The Embeddings Vector. For each Seq Element we have a Sentence, and for each sentence we have an Array for each of its words. Each of its words gets a float array to represent its Embeddings
    */
  def tag(batch: Seq[TokenizedSentence], embeddingsKey: String, dimension: Int): Seq[Array[Array[Float]]] = {

    val tensors = new TensorResources()

    /* Actual size of each sentence to skip padding in the TF model */
    val sequencesLength = batch.map(x => x.indexedTokens.length).toArray
    val maxSentenceLength = sequencesLength.max

    val sentencesBytes = batch.map { sentence =>

      val tokensArray = sentence.indexedTokens
      val diff = maxSentenceLength - tokensArray.length

      if(tokensArray.length < maxSentenceLength){
        val tokens = tokensArray.map{x=> x.token}
        /* Padding by adding extra empty tokens to smaller sentences */
        val newTokens = tokens ++ Array.fill(1, diff)(" ").head
        newTokens.map { token =>
          token.getBytes("UTF-8")
        }
      }else {
        tokensArray.map { token =>
          token.token.getBytes("UTF-8")
        }
      }

    }.toArray

    val sentenceTensors = tensors.createTensor(sentencesBytes)
    val runner = tensorflow.getSession(configProtoBytes = configProtoBytes).runner

    runner
      .feed(tokensKey, sentenceTensors)
      .feed(sequenceKey, tensors.createTensor(sequencesLength))
      .fetch(embeddingsKey)

    val outs = runner.run().asScala
    val wordEmbeddings = TensorResources.extractFloats(outs.head)
    tensors.clearSession(outs)
    tensors.clearTensors()

    var allWordEmbeddings: Array[Array[Float]] = wordEmbeddings.grouped(dimension).toArray
    /* Group embeddings based on the length of the sentence */
    val embeddingsBySentence = batch.map{ case (sentence) =>
      val embds = allWordEmbeddings.slice(0, sentence.indexedTokens.length)
      /* Remove the already used vectors */
      allWordEmbeddings = allWordEmbeddings.slice(0, sentence.indexedTokens.length*dimension)
      embds
    }

    batch.zip(embeddingsBySentence).map { case (ids, embeddings) =>
      embeddings
    }

  }

  /**
    * word_emb: the character-based word representations with shape [batch_size, max_length, 512].  == 512
    * lstm_outputs1: the first LSTM hidden state with shape [batch_size, max_length, 1024]. === 1024
    * lstm_outputs2: the second LSTM hidden state with shape [batch_size, max_length, 1024]. === 1024
    * elmo: the weighted sum of the 3 layers, where the weights are trainable. This tensor has shape [batch_size, max_length, 1024]  == 1024
    *
    * @param layer Layer specification
    * @return The dimension of chosen layer
    */
  def getDimensions(layer: String): Int = {
    {
      layer match {
        case "word_emb" =>
          512
        case "lstm_outputs1" =>
          1024
        case "lstm_outputs2" =>
          1024
        case "elmo" =>
          1024
      }
    }
  }
}