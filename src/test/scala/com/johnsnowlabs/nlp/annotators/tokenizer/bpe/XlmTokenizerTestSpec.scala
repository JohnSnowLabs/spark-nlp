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

import com.johnsnowlabs.nlp.annotators.common.TokenPiece
import org.scalatest.flatspec.AnyFlatSpec

class XlmTokenizerTestSpec extends AnyFlatSpec with BpeTokenizerBehaviours {
  val vocab: Map[String, Int] =
    Array(
      "<s>",
      "</s>",
      "<unk>",
      "<pad>",
      "<special0>",
      "<special1>",
      "<special2>",
      "<special3>",
      "<special4>",
      "<special5>",
      "<special6>",
      "<special7>",
      "<special8>",
      "<special9>",
      "i</w>",
      "un",
      "ambi",
      "gou",
      "os",
      "ly</w>",
      "good</w>",
      "3",
      "as",
      "d</w>",
      "!</w>").zipWithIndex.toMap

  val merges: Map[(String, String), Int] = Array(
    "u n",
    "a m",
    "a s",
    "o u",
    "b i",
    "o s",
    "i g",
    "n a",
    "am b",
    "g o",
    "s l",
    "o d</w>",
    "l y</w>",
    "am bi",
    "g ou",
    "m b",
    "go od</w>",
    "bi g",
    "s d</w>",
    "o o").map(_.split(" ")).map { case Array(c1, c2) => (c1, c2) }.zipWithIndex.toMap

  val modelType = "xlm"

  override def assertEncodedCorrectly(
      text: String,
      encoded: Array[TokenPiece],
      expected: Array[String],
      expectedIds: Array[Int]): Unit = {
    assert(encoded.length == expected.length)

    for (i <- encoded.indices) {
      val piece = encoded(i)
      assert(piece.wordpiece == expected(i))
      assert(piece.pieceId == expectedIds(i))

      // Lowercase, as xlm transforms the words into lowercase
      assert(text.slice(piece.begin, piece.end + 1).toLowerCase == piece.wordpiece)
    }
  }

  "XlmTokenizer" should behave like correctBpeTokenizer(
    text = "I unambigouosly good 3Asd!",
    expected = Array("i", "un", "ambi", "gou", "os", "ly", "good", "3", "as", "d", "!"),
    expectedIds = Array(14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24))

  it should behave like correctBpeTokenizerInFringeSituations()

  it should behave like correctBpeTokenizerSpecialTokens(
    text = "I unambigouosly <special1> good 3Asd <special1>",
    expected = Array(
      "i",
      "un",
      "ambi",
      "gou",
      "os",
      "ly",
      "<special1>",
      "good",
      "3",
      "as",
      "d",
      "<special1>"),
    expectedIds = Array(14, 15, 16, 17, 18, 19, 5, 20, 21, 22, 23, 5))
}
