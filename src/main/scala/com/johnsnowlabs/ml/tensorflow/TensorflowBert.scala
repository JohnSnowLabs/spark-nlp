package com.johnsnowlabs.ml.tensorflow

import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import com.johnsnowlabs.nlp.annotators.common._

import scala.collection.JavaConverters._

/**
  * BERT (Bidirectional Encoder Representations from Transformers) provides dense vector representations for natural language by using a deep, pre-trained neural network with the Transformer architecture
  *
  *
  * See [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/BertEmbeddingsTestSpec.scala]] for further reference on how to use this API.
  * Sources:
  *
  *
  * @param tensorflow           Bert Model wrapper with TensorFlow Wrapper
  * @param sentenceStartTokenId Id of sentence start Token
  * @param sentenceEndTokenId   Id of sentence end Token.
  * @param configProtoBytes     Configuration for TensorFlow session
  *
  *                             Paper:  [[ https://arxiv.org/abs/1810.04805]]
  *
  *                             Source:  [[https://github.com/google-research/bert]]
  **/
class TensorflowBert(val tensorflow: TensorflowWrapper,
                     sentenceStartTokenId: Int,
                     sentenceEndTokenId: Int,
                     configProtoBytes: Option[Array[Byte]] = None
                    ) extends Serializable {

  private val tokenIdsKey = "input_ids:0"
  private val maskIdsKey = "input_mask:0"
  private val segmentIdsKey = "segment_ids:0"
  private val embeddingsKey = "sequence_output:0"
  private val senteneEmbeddingsKey = "pooled_output:0"

  def encode(sentences: Seq[(WordpieceTokenizedSentence, Int)], maxSequenceLength: Int): Seq[Array[Int]] = {
    val maxSentenceLength = Array(maxSequenceLength - 2, sentences.map(_._1.tokens.length).max).min


    sentences.map(sentence => {
      val tokenPieceIds = sentence._1.tokens.map(t => t.pieceId)
      val padding = Array.fill(maxSentenceLength - tokenPieceIds.length)(0)

      Array(sentenceStartTokenId) ++
        tokenPieceIds.take(maxSentenceLength) ++
        Array(sentenceEndTokenId) ++
        padding
    })
  }

  def tag(batch: Seq[Array[Int]]): Seq[Array[Array[Float]]] = {
    val tensors = new TensorResources()
    val tensorsMasks = new TensorResources()
    val tensorsSegments = new TensorResources()

    val maxSentenceLength = batch.map(x => x.length).max
    val batchLength = batch.length

    val tokenBuffers = tensors.createIntBuffer(batchLength*maxSentenceLength)
    val maskBuffers = tensorsMasks.createIntBuffer(batchLength*maxSentenceLength)
    val segmentBuffers = tensorsSegments.createIntBuffer(batchLength*maxSentenceLength)

    val shape = Array(batch.length.toLong, maxSentenceLength)

    batch.map { sentence =>
      tokenBuffers.put(sentence)
      maskBuffers.put(sentence.map(x=> if (x == 0) 0 else 1))
      segmentBuffers.put(Array.fill(maxSentenceLength)(0))
    }

    tokenBuffers.flip()
    maskBuffers.flip()
    segmentBuffers.flip()

    val runner = tensorflow.getTFHubSession(configProtoBytes = configProtoBytes, initAllTables = false).runner

    val tokenTensors = tensors.createIntBufferTensor(shape, null)
    val maskTensors = tensorsMasks.createIntBufferTensor(shape, null)
    val segmentTensors = tensorsSegments.createIntBufferTensor(shape, null)

    runner
      .feed(tokenIdsKey, tokenTensors)
      .feed(maskIdsKey, maskTensors)
      .feed(segmentIdsKey, segmentTensors)
      .fetch(embeddingsKey)

    val outs = runner.run().asScala
    val embeddings = TensorResources.extractFloats(outs.head)

    tensors.clearSession(outs)
    tensors.clearTensors()
    tokenBuffers.clear()

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

  def tagSentence(batch: Seq[Array[Int]]): Array[Array[Float]] = {
    val tensors = new TensorResources()
    val tensorsMasks = new TensorResources()
    val tensorsSegments = new TensorResources()

    val maxSentenceLength = batch.map(x => x.length).max
    val batchLength = batch.length

    val tokenBuffers = tensors.createIntBuffer(batchLength*maxSentenceLength)
    val maskBuffers = tensorsMasks.createIntBuffer(batchLength*maxSentenceLength)
    val segmentBuffers = tensorsSegments.createIntBuffer(batchLength*maxSentenceLength)

    val shape = Array(batchLength.toLong, maxSentenceLength)

    batch.map { sentence =>
      tokenBuffers.put(sentence)
      maskBuffers.put(sentence.map(x=> if (x == 0) 0 else 1))
      segmentBuffers.put(Array.fill(maxSentenceLength)(0))
    }

    tokenBuffers.flip()
    maskBuffers.flip()
    segmentBuffers.flip()

    val runner = tensorflow.getTFHubSession(configProtoBytes = configProtoBytes, initAllTables = false).runner

    val tokenTensors = tensors.createIntBufferTensor(shape, null)
    val maskTensors = tensorsMasks.createIntBufferTensor(shape, null)
    val segmentTensors = tensorsSegments.createIntBufferTensor(shape, null)

    runner
      .feed(tokenIdsKey, tokenTensors)
      .feed(maskIdsKey, maskTensors)
      .feed(segmentIdsKey, segmentTensors)
      .fetch(senteneEmbeddingsKey)

    val outs = runner.run().asScala
    val embeddings = TensorResources.extractFloats(outs.head)

    tensors.clearSession(outs)
    tensors.clearTensors()
    tokenBuffers.clear()

    val dim = embeddings.length / batchLength
    embeddings.grouped(dim).toArray

  }

  def tagSentenceSBert(batch: Seq[Array[Int]]): Array[Array[Float]] = {
    val tensors = new TensorResources()
    val tensorsMasks = new TensorResources()
    val tensorsSegments = new TensorResources()

    val maxSentenceLength = batch.map(x => x.length).max
    val batchLength = batch.length

    val tokenBuffers = tensors.createLongBuffer(batchLength*maxSentenceLength)
    val maskBuffers = tensorsMasks.createLongBuffer(batchLength*maxSentenceLength)
    val segmentBuffers = tensorsSegments.createLongBuffer(batchLength*maxSentenceLength)

    val shape = Array(batchLength.toLong, maxSentenceLength)

    batch.map { sentence =>
      tokenBuffers.put(sentence.map(_.toLong))
      maskBuffers.put(sentence.map(x=> if (x == 0L) 0L else 1L))
      segmentBuffers.put(Array.fill(maxSentenceLength)(0L))
    }

    tokenBuffers.flip()
    maskBuffers.flip()
    segmentBuffers.flip()

    val runner = tensorflow.getTFHubSession(configProtoBytes = configProtoBytes, initAllTables = false).runner

    val tokenTensors = tensors.createLongBufferTensor(shape, null)
    val maskTensors = tensorsMasks.createLongBufferTensor(shape, null)
    val segmentTensors = tensorsSegments.createLongBufferTensor(shape, null)

    runner
      .feed(tokenIdsKey, tokenTensors)
      .feed(maskIdsKey, maskTensors)
      .feed(segmentIdsKey, segmentTensors)
      .fetch(senteneEmbeddingsKey)

    val outs = runner.run().asScala
    val embeddings = TensorResources.extractFloats(outs.head)

    tensors.clearSession(outs)
    tensors.clearTensors()
    tokenBuffers.clear()

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
    sentences.zipWithIndex.grouped(batchSize).flatMap{batch =>
      val encoded = encode(batch, maxSentenceLength)
      val vectors = tag(encoded)

      /*Combine tokens and calculated embeddings*/
      batch.zip(vectors).map{case (sentence, tokenVectors) =>
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

  def calculateSentenceEmbeddings(tokens: Seq[WordpieceTokenizedSentence],
                                  sentences: Seq[Sentence],
                                  batchSize: Int,
                                  maxSentenceLength: Int,
                                  isLong: Boolean = false
                                 ): Seq[Annotation] = {

    /*Run embeddings calculation by batches*/
    tokens.zipWithIndex.grouped(batchSize).flatMap{batch =>
      val encoded = encode(batch, maxSentenceLength)
      val embeddings = if (isLong) {
        tagSentenceSBert(encoded)
      } else {
        tagSentence(encoded)
      }

      sentences.zip(embeddings).map { case (sentence, vectors) =>
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

