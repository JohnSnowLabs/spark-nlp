package com.johnsnowlabs.ml.tensorflow

import com.johnsnowlabs.nlp.annotators.common.{IndexedToken, TokenPiece, TokenizedSentence, WordpieceTokenizedSentence}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}

trait TensorflowTokenClassification {

  protected val sentencePadTokenId: Int
  protected val sentenceStartTokenId: Int
  protected val sentenceEndTokenId: Int

  def predict(tokenizedSentences: Seq[TokenizedSentence], batchSize: Int, maxSentenceLength: Int,
              caseSensitive: Boolean, tags: Map[String, Int]): Seq[Annotation] = {

    val wordPieceTokenizedSentences = tokenizeWithAlignment(tokenizedSentences, maxSentenceLength, caseSensitive)

    /*Run calculation by batches*/
    wordPieceTokenizedSentences.zipWithIndex.grouped(batchSize).flatMap { batch =>
      val encoded = encode(batch, maxSentenceLength)
      val logits = tag(encoded)

      /*Combine tokens and calculated logits*/
      batch.zip(logits).flatMap { case (sentence, tokenVectors) =>
        val tokenLength = sentence._1.tokens.length

        /*All wordpiece logits*/
        val tokenLogits: Array[Array[Float]] = tokenVectors.slice(1, tokenLength + 1)

        val labelsWithScores = wordAndSpanLevelAlignmentWithTokenizer(tokenLogits, tokenizedSentences, sentence, tags)
        labelsWithScores
      }
    }.toSeq

  }

  def tokenizeWithAlignment(sentences: Seq[TokenizedSentence], maxSeqLength: Int, caseSensitive: Boolean): Seq[WordpieceTokenizedSentence]

  /** Encode the input sequence to indexes IDs adding padding where necessary
   * */
  def encode(sentences: Seq[(WordpieceTokenizedSentence, Int)], maxSequenceLength: Int): Seq[Array[Int]] = {
    //aka prepareBatchInputs
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

  def calculateSoftmax(scores: Array[Float]): Array[Float] = {
    val exp = scores.map(x => math.exp(x))
    exp.map(x => x / exp.sum).map(_.toFloat)
  }

  /** Word-level and span-level alignment with Tokenizer
    https://github.com/google-research/bert#tokenization

    ### Input
    orig_tokens = ["John", "Johanson", "'s",  "house"]
    labels      = ["NNP",  "NNP",      "POS", "NN"]

    # bert_tokens == ["[CLS]", "john", "johan", "##son", "'", "s", "house", "[SEP]"]
    # orig_to_tok_map == [1, 2, 4, 6]
   */
  def wordAndSpanLevelAlignmentWithTokenizer(tokenLogits: Array[Array[Float]], tokenizedSentences: Seq[TokenizedSentence],
                                             sentence: (WordpieceTokenizedSentence, Int), tags: Map[String, Int]): Seq[Annotation] = {

    val labelsWithScores = sentence._1.tokens.zip(tokenLogits).flatMap {
      case (tokenPiece, scores) =>
        val indexedToken = findIndexedToken(tokenizedSentences, sentence, tokenPiece)
        indexedToken.map {
          token =>
            val label = tags.find(_._2 == scores.zipWithIndex.maxBy(_._1)._2).map(_._1).getOrElse("NA")
            val meta = scores.zipWithIndex.flatMap(x => Map(tags.find(_._2 == x._2).map(_._1).toString -> x._1.toString))
            Annotation(
              annotatorType = AnnotatorType.NAMED_ENTITY,
              begin = token.begin,
              end = token.end,
              result = label,
              metadata = Map("sentence" -> sentence._2.toString, "word" -> token.token) ++ meta
            )
        }
    }
    labelsWithScores.toSeq
  }

  def findIndexedToken(tokenizedSentences: Seq[TokenizedSentence], sentence: (WordpieceTokenizedSentence, Int),
                       tokenPiece: TokenPiece): Option[IndexedToken]

}
