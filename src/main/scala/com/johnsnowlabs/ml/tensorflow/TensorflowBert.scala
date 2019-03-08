package com.johnsnowlabs.ml.tensorflow

import com.johnsnowlabs.nlp.annotators.common._

class TensorflowBert(tensorflow: TensorflowWrapper,
                     sentenceStartTokenId: Int,
                     sentenceEndTokenId: Int,
                     maxSentenceLength: Int,
                     batchSize: Int = 5
                    ) {

  private val tokenIdsKey = "token_ids:0"
  private val embeddingsKey = "bert/embeddings/LayerNorm/batchnorm/add_1:0"

  def encode(sentence: WordpieceTokenizedSentence): Array[Int] = {
    val tokens = sentence.tokens.map(t => t.pieceId)

    require(tokens.length <= maxSentenceLength - 2)
    Array(sentenceStartTokenId) ++
      tokens ++
      Array(sentenceEndTokenId) ++
      Array.fill(maxSentenceLength - tokens.length - 2)(0)
  }

  def tag(batch: Seq[Array[Int]]): Seq[Array[Array[Float]]] = {
    val tensors = new TensorResources()

    println(s"shape = ${batch.length}, ${batch(0).length}")

    val calculated = tensorflow.session.runner
      .feed(tokenIdsKey, tensors.createTensor(batch.toArray))
      .fetch(embeddingsKey)
      .run()

    tensors.clearTensors()

    val embeddings = TensorResources.extractFloats(calculated.get(0))

    val dim = embeddings.length / (batch.length * maxSentenceLength)
    val result = embeddings.grouped(dim).toArray.grouped(maxSentenceLength).toArray

    result
  }

  def calculateEmbeddings(sentences: Seq[WordpieceTokenizedSentence]): Seq[WordpieceEmbeddingsSentence] = {
    // ToDo What to do with longer sentences?

    // Run embeddings calculation by batches
    sentences.zipWithIndex.grouped(batchSize).flatMap{batch =>
      val encoded = batch.map(s => encode(s._1))
      val vectors = tag(encoded)

      // Combine tokens and calculated embeddings
      batch.zip(vectors).map{case (sentence, tokenVectors) =>
          val tokenLength = sentence._1.tokens.length
          // Sentence Embeddings are at first place (token [CLS]
          val sentenceEmbeddings = tokenVectors.headOption

          // All wordpiece embeddings
          val tokenEmbeddings = tokenVectors.slice(1, tokenLength + 1)

          // Leave embeddings only for word start
          val tokensWithEmbeddings = sentence._1.tokens.zip(tokenEmbeddings).flatMap{
            case (token, tokenEmbedding) =>
              if (token.isWordStart) {
                val tokenWithEmbeddings = TokenPieceEmbeddings(token, tokenEmbedding)
                Some(tokenWithEmbeddings)
              }
              else
                None
          }

        WordpieceEmbeddingsSentence(tokensWithEmbeddings, sentence._2, sentenceEmbeddings)
      }
    }.toSeq
  }
}
