package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.annotators.common._

trait TransformerEmbeddings {

  protected val sentencePadTokenId: Int
  protected val sentenceStartTokenId: Int
  protected val sentenceEndTokenId: Int

  def predict(tokenizedSentences: Seq[TokenizedSentence], batchSize: Int, maxSentenceLength: Int,
              caseSensitive: Boolean): Seq[WordpieceEmbeddingsSentence] = {

    //Tokenize sentences
    val wordPieceTokenizedSentences = tokenizeWithAlignment(tokenizedSentences, caseSensitive, maxSentenceLength)

    /*Run embeddings calculation by batches*/
    wordPieceTokenizedSentences.zipWithIndex.grouped(batchSize).flatMap { batch =>
      val encoded = encode(batch, maxSentenceLength)
      val vectors = tag(encoded)

      /*Combine tokens and calculated embeddings*/
      batch.zip(vectors).map { case (sentence, tokenVectors) =>
        val tokenLength = sentence._1.tokens.length

        /*All wordpiece embeddings*/
        val tokenEmbeddings = tokenVectors.slice(1, tokenLength + 1)

        val wordpieceEmbeddingsSentence = wordAndSpanLevelAlignmentWithTokenizer(tokenEmbeddings, caseSensitive,
          tokenizedSentences, sentence)
        wordpieceEmbeddingsSentence
      }
    }.toSeq

  }

  def tokenizeWithAlignment(tokenizedSentences: Seq[TokenizedSentence], caseSensitive: Boolean,
                            maxSentenceLength: Int): Seq[WordpieceTokenizedSentence]

  /** Encode the input sequence to indexes IDs adding padding where necessary */
  def encode(sentences: Seq[(WordpieceTokenizedSentence, Int)], maxSequenceLength: Int): Seq[Array[Int]] = {
    val maxSentenceLength =
      Array(
        maxSequenceLength - 2,
        sentences.map { case (wpTokSentence, _) => wpTokSentence.tokens.length }.max).min

    sentences
      .map { case (wpTokSentence, _) =>
        val tokenPieceIds = wpTokSentence.tokens.map(t => t.pieceId)
        val padding = Array.fill(maxSentenceLength - tokenPieceIds.length)(sentencePadTokenId)

        Array(sentenceStartTokenId) ++ tokenPieceIds.take(maxSentenceLength) ++ Array(sentenceEndTokenId) ++ padding
      }
  }

  def tag(batch: Seq[Array[Int]]): Seq[Array[Array[Float]]]

  /** Word-level and span-level alignment with Tokenizer
   * https://github.com/google-research/bert#tokenization
   *
   * ### Input
   * orig_tokens = ["John", "Johanson", "'s",  "house"]
   * labels      = ["NNP",  "NNP",      "POS", "NN"]
   *
   * # bert_tokens == ["[CLS]", "john", "johan", "##son", "'", "s", "house", "[SEP]"]
   * # orig_to_tok_map == [1, 2, 4, 6]
   */
  def wordAndSpanLevelAlignmentWithTokenizer(tokenEmbeddings: Array[Array[Float]], caseSensitive: Boolean,
                                             tokenizedSentences: Seq[TokenizedSentence],
                                             sentence: (WordpieceTokenizedSentence, Int)): WordpieceEmbeddingsSentence = {
    val tokensWithEmbeddings = sentence._1.tokens.zip(tokenEmbeddings).flatMap {
      case (token, tokenEmbedding) =>
        val tokenWithEmbeddings = TokenPieceEmbeddings(token, tokenEmbedding)
        val originalTokensWithEmbeddings = tokenizedSentences(sentence._2).indexedTokens.find(
          p => p.begin == tokenWithEmbeddings.begin && tokenWithEmbeddings.isWordStart).map {
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

}
