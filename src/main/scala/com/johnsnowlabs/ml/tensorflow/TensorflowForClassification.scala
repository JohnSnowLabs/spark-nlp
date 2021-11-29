/*
 * Copyright 2017-2021 John Snow Labs
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

import com.johnsnowlabs.nlp.annotators.common.{IndexedToken, Sentence, TokenPiece, TokenizedSentence, WordpieceTokenizedSentence}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}

trait TensorflowForClassification {

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

  def predictSequence(tokenizedSentences: Seq[TokenizedSentence], sentences: Seq[Sentence], batchSize: Int, maxSentenceLength: Int,
                      caseSensitive: Boolean, coalesceSentences: Boolean = false, tags: Map[String, Int]): Seq[Annotation] = {

    val wordPieceTokenizedSentences = tokenizeWithAlignment(tokenizedSentences, maxSentenceLength, caseSensitive)

    /*Run calculation by batches*/
    wordPieceTokenizedSentences.zip(sentences).zipWithIndex.grouped(batchSize).flatMap { batch =>
      val tokensBatch = batch.map(x => (x._1._1, x._2))
      val encoded = encode(tokensBatch, maxSentenceLength)
      val logits = tagSequence(encoded)

      if (coalesceSentences) {
        val scores = logits.transpose.map(_.sum / logits.length)
        val label = tags.find(_._2 == scores.zipWithIndex.maxBy(_._1)._2).map(_._1).getOrElse("NA")
        val meta = scores.zipWithIndex.flatMap(x => Map(tags.find(_._2 == x._2).map(_._1).toString -> x._1.toString))
        Array(
          Annotation(
            annotatorType = AnnotatorType.CATEGORY,
            begin = sentences.head.start,
            end = sentences.head.end,
            result = label,
            metadata = Map("sentence" -> sentences.head.index.toString) ++ meta
          )
        )
      } else {
        sentences.zip(logits).map { case (sentence, scores) =>
          val label = tags.find(_._2 == scores.zipWithIndex.maxBy(_._1)._2).map(_._1).getOrElse("NA")
          val meta = scores.zipWithIndex.flatMap(x => Map(tags.find(_._2 == x._2).map(_._1).toString -> x._1.toString))
          Annotation(
            annotatorType = AnnotatorType.CATEGORY,
            begin = sentence.start,
            end = sentence.end,
            result = label,
            metadata = Map("sentence" -> sentence.index.toString) ++ meta
          )
        }
      }
    }.toSeq

  }

  def tokenizeWithAlignment(sentences: Seq[TokenizedSentence], maxSeqLength: Int, caseSensitive: Boolean): Seq[WordpieceTokenizedSentence]

  /** Encode the input sequence to indexes IDs adding padding where necessary
   * */
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

  def tagSequence(batch: Seq[Array[Int]]): Array[Array[Float]]

  def calculateSoftmax(scores: Array[Float]): Array[Float] = {
    val exp = scores.map(x => math.exp(x))
    exp.map(x => x / exp.sum).map(_.toFloat)
  }

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
