/*
 * Copyright 2017-2022 John Snow Labs
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

package com.johnsnowlabs.nlp.annotators.tokenizer.bpe

import com.johnsnowlabs.nlp.annotators.common.{IndexedToken, Sentence, TokenPiece}
import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec

trait BpeTokenizerBehaviours {
  this: AnyFlatSpec =>

  val modelType: String
  val merges: Map[(String, String), Int]
  val vocab: Map[String, Int]
  val replaceCharBeforeAssertion: Option[String] = None

  lazy private val defaultTokenizer: BpeTokenizer =
    BpeTokenizer.forModel(modelType, merges, vocab)

  protected def assertEncodedCorrectly(
      text: String,
      encoded: Array[TokenPiece],
      expected: Array[String],
      expectedIds: Array[Int]): Unit = {

    assert(encoded.length == expected.length)
    for (i <- encoded.indices) {
      val piece = encoded(i)
      assert(piece.wordpiece == expected(i))
      assert(piece.pieceId == expectedIds(i))

      replaceCharBeforeAssertion match {
        case Some(decodeChar) =>
          assert(
            text.slice(piece.begin, piece.end + 1) == piece.wordpiece.replace(decodeChar, " "))
        case _ => assert(text.slice(piece.begin, piece.end + 1) == piece.wordpiece)
      }
    }
  }

  protected def tokenizeAndEncode(
      tokenizer: BpeTokenizer,
      text: String): (Array[IndexedToken], Array[TokenPiece]) = {
    val sentence = Sentence(text, 0, text.length - 1, 0)

    val tokenized = tokenizer.tokenize(sentence)
    val encoded = tokenizer.encode(tokenized)
    (tokenized, encoded)
  }

  def correctBpeTokenizer(
      text: String,
      expected: Array[String],
      expectedIds: Array[Int]): Unit = {

    it should "encode words correctly" taggedAs FastTest in {
      val (_, encoded: Array[TokenPiece]) = tokenizeAndEncode(defaultTokenizer, text)
      assertEncodedCorrectly(text, encoded, expected, expectedIds)
    }

    it should "add sentence padding correctly if requested" taggedAs FastTest in {
      val sentencePaddingTokenizer =
        BpeTokenizer.forModel(modelType, merges, vocab, padWithSequenceTokens = true)

      val (tokenized: Array[IndexedToken], encoded: Array[TokenPiece]) =
        tokenizeAndEncode(sentencePaddingTokenizer, text)

      assert(tokenized.head.token == sentencePaddingTokenizer.sentencePadding._1)
      assert(tokenized.last.token == sentencePaddingTokenizer.sentencePadding._2)

      assert(encoded.head.pieceId == sentencePaddingTokenizer.specialTokens.sentenceStart.id)
      assert(encoded.last.pieceId == sentencePaddingTokenizer.specialTokens.sentenceEnd.id)
    }
  }

  def correctBpeTokenizerWithAddedPrefixSpace(
      text: String,
      expected: Array[String],
      expectedIds: Array[Int]): Unit = {
    it should "encode words correctly with added prefix" taggedAs FastTest in {
      val addedPrefixTokenizer =
        BpeTokenizer.forModel(modelType, merges, vocab, addPrefixSpaceToSentence = true)

      val (_, encoded: Array[TokenPiece]) = tokenizeAndEncode(addedPrefixTokenizer, text)
      assertEncodedCorrectly(text, encoded, expected, expectedIds)
    }
  }

  def correctBpeTokenizerInFringeSituations(unknownTokenText: String = "???"): Unit = {
    it should "handle empty sentences" taggedAs FastTest in {
      val text = " \n"
      val (tokenized: Array[IndexedToken], encoded: Array[TokenPiece]) =
        tokenizeAndEncode(defaultTokenizer, text)
      assert(tokenized.isEmpty)
      assert(encoded.isEmpty)
    }

    it should "handle unknown words" taggedAs FastTest in {
      val text = unknownTokenText
      val (_, encoded: Array[TokenPiece]) = tokenizeAndEncode(defaultTokenizer, text)

      assert(encoded.forall(_.pieceId == defaultTokenizer.specialTokens.unk.id))
    }

    it should "handle unknown byte encodings" taggedAs FastTest in {
      val text = "I unambigouosly \u8216"
      val (_, encoded: Array[TokenPiece]) = tokenizeAndEncode(defaultTokenizer, text)

      assert(encoded.last.pieceId == defaultTokenizer.specialTokens.unk.id)
    }
  }

  def correctBpeTokenizerSpecialTokens(
      text: String,
      expected: Array[String],
      expectedIds: Array[Int]): Unit = {

    it should "encode special tokens correctly" taggedAs FastTest in {
      val (_, encoded: Array[TokenPiece]) = tokenizeAndEncode(defaultTokenizer, text)
      assertEncodedCorrectly(text, encoded, expected, expectedIds)
    }
  }
}
