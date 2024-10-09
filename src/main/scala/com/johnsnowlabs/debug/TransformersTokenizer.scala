package com.johnsnowlabs.debug

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.annotators.common.{IndexedToken, Sentence, TokenPiece, TokenizedSentence, WordpieceTokenizedSentence}
import com.johnsnowlabs.nlp.annotators.tokenizer.wordpiece.{BasicTokenizer, WordpieceEncoder}

class TransformersTokenizer(
    val sentenceStartToken: String,
    val sentencePadToken: String,
    val sentenceEndToken: String,
    vocabulary: Map[String, Int]) {

  private val sentenceStartTokenId = vocabulary(sentenceStartToken)
  private val sentencePadTokenId = vocabulary(sentencePadToken)
  private val sentenceEndTokenId = vocabulary(sentenceEndToken)

  def tokenizeWithAlignment(
      sentences: Seq[TokenizedSentence],
      maxSeqLength: Int,
      caseSensitive: Boolean): Seq[WordpieceTokenizedSentence] = {

    val basicTokenizer = new BasicTokenizer(caseSensitive)
    val encoder = new WordpieceEncoder(vocabulary)

    sentences.map { tokenIndex =>
      // filter empty and only whitespace tokens
      val bertTokens =
        tokenIndex.indexedTokens.filter(x => x.token.nonEmpty && !x.token.equals(" ")).map {
          token =>
            val content = if (caseSensitive) token.token else token.token.toLowerCase()
            val sentenceBegin = token.begin
            val sentenceEnd = token.end
            val sentenceIndex = tokenIndex.sentenceIndex
            val result = basicTokenizer.tokenize(
              Sentence(content, sentenceBegin, sentenceEnd, sentenceIndex))
            if (result.nonEmpty) result.head else IndexedToken("")
        }
      val wordpieceTokens = bertTokens.flatMap(token => encoder.encode(token)).take(maxSeqLength)
      WordpieceTokenizedSentence(wordpieceTokens)
    }
  }

  def tokenizeDocument(
                        docs: Seq[Annotation],
                        maxSeqLength: Int,
                        caseSensitive: Boolean): Seq[WordpieceTokenizedSentence] = {

    // we need the original form of the token
    // let's lowercase if needed right before the encoding
    val basicTokenizer = new BasicTokenizer(caseSensitive = true, hasBeginEnd = false)
    val encoder = new WordpieceEncoder(vocabulary)
    val sentences = docs.map { s => Sentence(s.result, s.begin, s.end, 0) }

    sentences.map { sentence =>
      val tokens = basicTokenizer.tokenize(sentence)

      val wordpieceTokens = if (caseSensitive) {
        tokens.flatMap(token => encoder.encode(token))
      } else {
        // now we can lowercase the tokens since we have the original form already
        val normalizedTokens =
          tokens.map(x => IndexedToken(x.token.toLowerCase(), x.begin, x.end))
        val normalizedWordPiece = normalizedTokens.flatMap(token => encoder.encode(token))

        normalizedWordPiece.map { t =>
          val orgToken = tokens
            .find(org => t.begin == org.begin && t.isWordStart)
            .map(x => x.token)
            .getOrElse(t.token)
          TokenPiece(t.wordpiece, orgToken, t.pieceId, t.isWordStart, t.begin, t.end)
        }
      }

      WordpieceTokenizedSentence(wordpieceTokens)
    }
  }

  /** Encode the input sequence to indexes IDs adding padding where necessary */
  def encode(
      sentences: Seq[(WordpieceTokenizedSentence, Int)],
      maxSequenceLength: Int): Seq[Array[Int]] = {
    val maxSentenceLength =
      Array(
        maxSequenceLength - 2,
        sentences.map { case (wpTokSentence, _) =>
          wpTokSentence.tokens.length
        }.max).min

    sentences
      .map { case (wpTokSentence, _) =>
        val tokenPieceIds = wpTokSentence.tokens.map(t => t.pieceId)
        val padding = Array.fill(maxSentenceLength - tokenPieceIds.length)(sentencePadTokenId)

        Array(sentenceStartTokenId) ++ tokenPieceIds.take(maxSentenceLength) ++ Array(
          sentenceEndTokenId) ++ padding
      }
  }

  def encodeSequenceRegular(
                      seq1: Seq[WordpieceTokenizedSentence],
                      seq2: Seq[WordpieceTokenizedSentence],
                      maxSequenceLength: Int): Seq[Array[Int]] = {

    val question = seq1
      .flatMap { wpTokSentence =>
        wpTokSentence.tokens.map(t => t.pieceId)
      }
      .toArray
      .take(maxSequenceLength - 2) ++ Array(sentenceEndTokenId)

    val context = seq2
      .flatMap { wpTokSentence =>
        wpTokSentence.tokens.map(t => t.pieceId)
      }
      .toArray
      .take(maxSequenceLength - question.length - 2) ++ Array(sentenceEndTokenId)

    Seq(Array(sentenceStartTokenId) ++ question ++ context)
  }

  def encodeSequence(
                      seq1: Seq[WordpieceTokenizedSentence],
                      seq2: Seq[WordpieceTokenizedSentence],
                      maxSequenceLength: Int): Seq[Array[Int]] = {

    // Convert WordpieceTokenizedSentence to a flat sequence of piece IDs
    val question = seq1.flatMap { wpTokSentence =>
      wpTokSentence.tokens.map(t => t.pieceId)
    }.toArray

    val context = seq2.flatMap { wpTokSentence =>
      wpTokSentence.tokens.map(t => t.pieceId)
    }.toArray

    // Total available space for tokens (excluding special tokens)
    val availableLength = maxSequenceLength - 3  // 1 for sentenceStartTokenId, 1 for question-end, 1 for context-end

    // Truncate the question if necessary
    val truncatedQuestion = question.take(availableLength)

    // Use the remaining space for the context
    val remainingLength = availableLength - truncatedQuestion.length
    val truncatedContext = context.take(remainingLength)

    // Assemble final sequence with special tokens
    val sequence = Array(sentenceStartTokenId) ++ truncatedQuestion ++ Array(sentenceEndTokenId) ++ truncatedContext ++ Array(sentenceEndTokenId)

    // Pad the sequence if it's shorter than maxSequenceLength
    val paddingLength = maxSequenceLength - sequence.length
    val paddedSequence = if (paddingLength > 0) {
      sequence ++ Array.fill(paddingLength)(0)  // Pad with 0 (or padding token)
    } else {
      sequence  // No padding needed
    }

    // Return the padded (or truncated) sequence
    Seq(paddedSequence)
  }





}
