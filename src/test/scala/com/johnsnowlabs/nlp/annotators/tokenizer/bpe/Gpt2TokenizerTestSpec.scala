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

package com.johnsnowlabs.nlp.annotators.tokenizer.bpe

import com.johnsnowlabs.nlp.annotators.common.{Sentence, TokenPiece}
import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec

class Gpt2TokenizerTestSpec extends AnyFlatSpec {
  val vocab: Map[String, Int] =
    Array(
      "<|endoftext|>",
      "ĠI",
      "Ġunamb",
      "ig",
      "ou",
      "os",
      "ly",
      "Ġgood",
      "Ġ3",
      "ĠAs",
      "d",
      "Ġ!"
    ).zipWithIndex.toMap

  val merges: Map[(String, String), Int] = Array(
    "o u",
    "l y",
    "Ġ g",
    "a m",
    "i g",
    "Ġ u",
    "o d",
    "u n",
    "o s",
    "Ġg o",
    "Ġu n",
    "o od",
    "A s",
    "m b",
    "g o",
    "o o",
    "n a",
    "am b",
    "s l",
    "n am",
    "b i",
    "b ig",
    "u o",
    "s d",
    "Ġun amb",
    "Ġgo od",
    "Ġ 3"
  ).map(_.split(" ")).map { case Array(c1, c2) => (c1, c2) }.zipWithIndex.toMap

  val bpeTokenizer: BpeTokenizer = BpeTokenizer.forModel(
    "gpt2",
    merges,
    vocab
  )

  private def assertEncodedCorrectly(text: String,
                                     encoded: Array[TokenPiece],
                                     expected: Array[String],
                                     expectedIds: Array[Int]): Unit = {
    println(encoded.map {
      t: TokenPiece => t.wordpiece
    }.mkString("Array(", ", ", ")"))
    println(encoded.map {
      t: TokenPiece => t.pieceId
    }.mkString("Array(", ", ", ")"))
    assert(encoded.length == expected.length)
    for (i <- encoded.indices) {
      val piece = encoded(i)
      assert(piece.wordpiece == expected(i))
      assert(piece.pieceId == expectedIds(i))

      assert(text.slice(piece.begin, piece.end + 1) == piece.wordpiece.replace("Ġ", " "))
    }
  }

  it should "encode words correctly" taggedAs FastTest in {
    val text = "I unambigouosly good 3Asd!"
    val sentence = Sentence(text, 0, text.length - 1, 0)

    val expected: Array[String] = Array("I", "Ġunamb", "ig", "ou", "os", "ly", "Ġgood", "Ġ3", "As", "d", "!")
    val expectedIds: Array[Int] = Array(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)

    val tokenized = bpeTokenizer.tokenize(sentence)
    val encoded = bpeTokenizer.encode(tokenized)

    assertEncodedCorrectly(text, encoded, expected, expectedIds)

  }

  it should "encode sentences with special tokens correctly" taggedAs FastTest in {
    val text = "I unambigouosly good 3Asd <|endoftext|>"
    val sentence = Sentence(text, 0, text.length - 1, 0)

    val expected = Array("I", "Ġunamb", "ig", "ou", "os", "ly", "Ġgood", "Ġ3", "As", "d", "<|endoftext|>")
    val expectedIds = Array(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0)

    val tokenizedWithMask = bpeTokenizer.tokenize(sentence)
    val encoded = bpeTokenizer.encode(tokenizedWithMask)

    assertEncodedCorrectly(text, encoded, expected, expectedIds)
  }

  it should "handle empty sentences" taggedAs FastTest in {
    val text = " \n"
    val sentence = Sentence(text, 0, text.length - 1, 0)

    val tokenized = bpeTokenizer.tokenize(sentence)
    val encoded = bpeTokenizer.encode(tokenized)
    assert(tokenized.isEmpty)
    assert(encoded.isEmpty)
  }

  it should "add sentence padding correctly if requested" taggedAs FastTest in {
    val tokenizer = BpeTokenizer.forModel("gpt2", merges, vocab, padWithSentenceTokens = true)

    val text = "I unambigouosly"
    val sentence = Sentence(text, 0, text.length - 1, 0)

    val expected = Array("<|endoftext|>", "I", "Ġunamb", "ig", "ou", "os", "ly", "<|endoftext|>")
    val expectedIds = Array(0, 1, 2, 3, 4, 5, 6, 0)

    val tokenized = tokenizer.tokenize(sentence)
    val encoded = tokenizer.encode(tokenized)

    val textPadded = "<|endoftext|>I unambigouosly<|endoftext|>"
    assertEncodedCorrectly(textPadded, encoded, expected, expectedIds)

    assert(tokenized.head.token == tokenizer.sentencePadding._1)
    assert(tokenized.last.token == tokenizer.sentencePadding._2)

  }
  it should "handle unknown words" taggedAs FastTest in {
    val text = "???"
    val sentence = Sentence(text, 0, text.length - 1, 0)

    val tokenized = bpeTokenizer.tokenize(sentence)
    val encoded = bpeTokenizer.encode(tokenized)
    assert(encoded.forall(_.pieceId == vocab("<|endoftext|>")))
  }

  it should "handle unknown byte encodings" taggedAs FastTest in {
    val text = "I unambigouosly \u8216"
    val sentence = Sentence(text, 0, text.length - 1, 0)

    val tokenized = bpeTokenizer.tokenize(sentence)
    val encoded = bpeTokenizer.encode(tokenized)
    assert(encoded.last.pieceId == vocab("<|endoftext|>"))
  }
}
