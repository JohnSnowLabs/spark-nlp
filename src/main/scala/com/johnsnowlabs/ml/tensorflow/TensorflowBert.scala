package com.johnsnowlabs.ml.tensorflow

import com.johnsnowlabs.nlp.annotators.common._
import scala.collection.JavaConverters._


class TensorflowBert(val tensorflow: TensorflowWrapper,
                     sentenceStartTokenId: Int,
                     sentenceEndTokenId: Int,
                     configProtoBytes: Option[Array[Byte]] = None
                    ) extends Serializable {

  private val tokenIdsKey = "module/input_ids"
  private val maskIdsKey = "module/input_mask"
  private val segmentIdsKey = "module/segment_ids"

  def encode(sentence: WordpieceTokenizedSentence, maxSentenceLength: Int): Array[Int] = {
    val tokens = sentence.tokens.map(t => t.pieceId)

    Array(sentenceStartTokenId) ++
      tokens ++
      Array(sentenceEndTokenId) ++
      Array.fill(maxSentenceLength - tokens.length - 2)(0)
  }

  def tag(batch: Seq[Array[Int]], embeddingsKey: String, maxSentenceLength: Int): Seq[Array[Array[Float]]] = {
    val tensors = new TensorResources()
    val tensorsMasks = new TensorResources()
    val tensorsSegments = new TensorResources()

    val tokenBuffers = tensors.createIntBuffer(batch.length*maxSentenceLength)
    val maskBuffers = tensorsMasks.createIntBuffer(batch.length*maxSentenceLength)
    val segmentBuffers = tensorsSegments.createIntBuffer(batch.length*maxSentenceLength)

    val shape = Array(batch.length.toLong, maxSentenceLength)

    batch.map { sentence =>
      if (sentence.length > maxSentenceLength) {
        tokenBuffers.put(sentence.take(maxSentenceLength - 1) ++ Array(sentenceEndTokenId))
        maskBuffers.put(sentence.take(maxSentenceLength).map(x=> if (x == 0) 0 else 1))
        segmentBuffers.put(Array.fill(maxSentenceLength)(0))
      }
      else {
        tokenBuffers.put(sentence)
        maskBuffers.put(sentence.map(x=> if (x == 0) 0 else 1))
        segmentBuffers.put(Array.fill(maxSentenceLength)(0))
      }
    }

    tokenBuffers.flip()
    maskBuffers.flip()
    segmentBuffers.flip()

    val runner = tensorflow.getTFHubSession(configProtoBytes = configProtoBytes).runner

    val tokenTensors = tensors.createIntBufferTensor(shape, tokenBuffers)
    val maskTensors = tensorsMasks.createIntBufferTensor(shape, maskBuffers)
    val segmentTensors = tensorsSegments.createIntBufferTensor(shape, segmentBuffers)

    runner
      .feed(tokenIdsKey, tokenTensors)
      .feed(maskIdsKey, maskTensors)
      .feed(segmentIdsKey, segmentTensors)
      .fetch(embeddingsKey)
    //
    val outs = runner.run().asScala
    val embeddings = TensorResources.extractFloats(outs.head)
    //
    tensors.clearSession(outs)
    tensors.clearTensors()
    tokenBuffers.clear()

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

  def extractPoolingLayer(layer: Int, dimension: Int): String = {
    val bertLayer = if(dimension == 768){
      layer match {
        case -1 =>
          "module/bert/encoder/Reshape_13:0"
        case -2 =>
          "module/bert/encoder/Reshape_12:0"
        case 0 =>
          "module/bert/encoder/Reshape_1:0"
      }
    } else {
      layer match {
        case -1 =>
          "module/bert/encoder/Reshape_25:0"
        case -2 =>
          "module/bert/encoder/Reshape_24:0"
        case 0 =>
          "module/bert/encoder/Reshape_1:0"
      }
    }
    bertLayer
  }

  def calculateEmbeddings(sentences: Seq[WordpieceTokenizedSentence],
                          originalTokenSentences: Seq[TokenizedSentence],
                          poolingLayer: Int,
                          batchSize: Int,
                          maxSentenceLength: Int,
                          dimension: Int,
                          caseSensitive: Boolean
                         ): Seq[WordpieceEmbeddingsSentence] = {

    /*Run embeddings calculation by batches*/
    sentences.zipWithIndex.grouped(batchSize).flatMap{batch =>
      val encoded = batch.map(s => encode(s._1, maxSentenceLength))

      val vectors = tag(encoded, extractPoolingLayer(poolingLayer, dimension), maxSentenceLength)

      /*Combine tokens and calculated embeddings*/
      batch.zip(vectors).map{case (sentence, tokenVectors) =>
        originalTokenSentences.length
        val tokenLength = sentence._1.tokens.length

        /*All wordpiece embeddings*/
        val tokenEmbeddings = tokenVectors.slice(1, tokenLength + 1)

        /*Word-level and span-level alignment with Tokenizer
        https://github.com/google-research/bert#tokenization*/
        val tokensWithEmbeddings = sentence._1.tokens.zip(tokenEmbeddings).flatMap{
          case (token, tokenEmbedding) =>
            val tokenWithEmbeddings = TokenPieceEmbeddings(token, tokenEmbedding)
            val originalTokensWithEmbeddings = originalTokenSentences(sentence._2).indexedTokens.find(
              p => p.begin == tokenWithEmbeddings.begin).map{
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

        WordpieceEmbeddingsSentence(tokensWithEmbeddings, sentence._2)
      }
    }.toSeq
  }
}

