package com.johnsnowlabs.ml.tensorflow

import com.johnsnowlabs.nlp.annotators.common._

class TensorflowBert(val tensorflow: TensorflowWrapper,
                     sentenceStartTokenId: Int,
                     sentenceEndTokenId: Int,
                     maxSentenceLength: Int,
                     batchSize: Int,
                     val dimension: Int,
                     val caseSensitive: Boolean,
                     configProtoBytes: Option[Array[Byte]] = None
                    ) extends Serializable {

  private val tokenIdsKey = "token_ids:0"

  def encode(sentence: WordpieceTokenizedSentence): Array[Int] = {
    val tokens = sentence.tokens.map(t => t.pieceId)

    Array(sentenceStartTokenId) ++
      tokens ++
      Array(sentenceEndTokenId) ++
      Array.fill(maxSentenceLength - tokens.length - 2)(0)

  }

  def tag(batch: Seq[Array[Int]], embeddingsKey: String): Seq[Array[Array[Float]]] = {
    val tensors = new TensorResources()

    //println(s"shape = ${batch.length}, ${batch(0).length}")
    val shrink = batch.map { sentence =>
      if (sentence.length > maxSentenceLength) {
        sentence.take(maxSentenceLength - 1) ++ Array(sentenceEndTokenId)
      }
      else {
        sentence
      }
    }.toArray

    val calculated = tensorflow.getSession(configProtoBytes = configProtoBytes).runner
      .feed(tokenIdsKey, tensors.createTensor(shrink))
      .fetch(embeddingsKey)
      .run()

    tensors.clearTensors()

    val embeddings = TensorResources.extractFloats(calculated.get(0))

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

  def extractPoolingLayer(layer: Int): String = {
    val bertLayer = if(dimension == 768){
      layer match {
        case -1 =>
          "bert/encoder/Reshape_13:0"
        case -2 =>
          "bert/encoder/Reshape_12:0"
        case 0 =>
          "bert/encoder/Reshape_1:0"
      }
    } else {
      layer match {
        case -1 =>
          "bert/encoder/Reshape_25:0"
        case -2 =>
          "bert/encoder/Reshape_24:0"
        case 0 =>
          "bert/encoder/Reshape_1:0"
      }
    }
    bertLayer
  }

  def calculateEmbeddings(sentences: Seq[WordpieceTokenizedSentence], originalTokenSentences: Seq[TokenizedSentence], poolingLayer: Int): Seq[WordpieceEmbeddingsSentence] = {
    // ToDo What to do with longer sentences?

    val bertLayer = extractPoolingLayer(poolingLayer)

    // Run embeddings calculation by batches
    sentences.zipWithIndex.grouped(batchSize).flatMap{batch =>
      val encoded = batch.map(s => encode(s._1))

      val vectors = tag(encoded, bertLayer)

      // Combine tokens and calculated embeddings
      batch.zip(vectors).map{case (sentence, tokenVectors) =>
        originalTokenSentences.length
        val tokenLength = sentence._1.tokens.length
        // Sentence Embeddings are at first place (token [CLS]
        val sentenceEmbeddings = tokenVectors.headOption

        // All wordpiece embeddings
        val tokenEmbeddings = tokenVectors.slice(1, tokenLength + 1)

        // Word-level and span-level alignment with Tokenizer
        // https://github.com/google-research/bert#tokenization
        val tokensWithEmbeddings = sentence._1.tokens.zip(tokenEmbeddings).flatMap{
          case (token, tokenEmbedding) =>
            val tokenWithEmbeddings = TokenPieceEmbeddings(token, tokenEmbedding)
            val originalTokensWithEmbeddings = originalTokenSentences(sentence._2).indexedTokens.find(p => p.begin == tokenWithEmbeddings.begin).map{
              case (token) =>
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

        WordpieceEmbeddingsSentence(tokensWithEmbeddings, sentence._2, sentenceEmbeddings)
      }
    }.toSeq
  }
}
