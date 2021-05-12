/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
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

import com.johnsnowlabs.nlp.annotators.common.Sentence
import com.johnsnowlabs.tags.FastTest
import org.scalatest.FlatSpec

class BpeTokenizerTestSpec extends FlatSpec {

  val vocab: Map[String, Int] =
    Array(
      "<s>",
      "</s>",
      "<mask>",
      "I",
      "Ġunamb",
      "ig",
      "ou",
      "os",
      "ly",
      "Ġgood",
      "Ġ3",
      "As",
      "d",
      "!"
    ).zipWithIndex.toMap

  val merges: Array[String] = Array(
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
  )

  val bpeTokenizer = new BpeTokenizer(merges, vocab)

  /**
   * TODO Remove this example, this is for review only
   * Example Output:
   * {{{
   * Array(
   *   IndexedToken(I,0,1),
   *   IndexedToken( unambigouosly,1,15),
   *   IndexedToken( good,15,20),
   *   IndexedToken( 3,20,22),
   *   IndexedToken(Asd,22,25),
   *   IndexedToken(!,25,26)
   * )
   * Array(
   *   TokenPiece(I,I,3,true,0,1),
   *   TokenPiece(Ġunamb,Ġunambigouosly,4,true,1,7),
   *   TokenPiece(ig,Ġunambigouosly,5,false,7,9),
   *   TokenPiece(ou,Ġunambigouosly,6,false,9,11),
   *   TokenPiece(os,Ġunambigouosly,7,false,11,13),
   *   TokenPiece(ly,Ġunambigouosly,8,false,13,15),
   *   TokenPiece(Ġgood,Ġgood,9,true,15,20),
   *   TokenPiece(Ġ3,Ġ3,10,true,20,22),
   *   TokenPiece(As,Asd,11,true,22,24),
   *   TokenPiece(d,Asd,12,false,24,25),
   *   TokenPiece(!,!,13,true,25,26)
   * )
   * }}}
   */

  "BpeTokenizer" should "encode words correctly" taggedAs FastTest in {
    val text = "I unambigouosly good 3Asd!"
    val sentence = Sentence(text, 0, text.length - 1, 0)

    val expected = Array("I", "Ġunamb", "ig", "ou", "os", "ly", "Ġgood", "Ġ3", "As", "d", "!")
    val expectedIds = Array(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)

    val tokenized = bpeTokenizer.tokenize(sentence)
    val encoded = bpeTokenizer.encode(tokenized)
    for (i <- encoded.indices) {
      val piece = encoded(i)
      assert(piece.wordpiece == expected(i))
      assert(piece.pieceId == expectedIds(i))

      assert(text.slice(piece.begin, piece.end) == piece.wordpiece.replace("Ġ", " "))
    }
  }

  "BpeTokenizer" should "encode sentences with special tokens correctly" taggedAs FastTest in {
    val text = "I unambigouosly <mask> 3Asd!"

    val sentence = Sentence(text, 0, text.length - 1, 0)
    val expected = Array("I", "Ġunamb", "ig", "ou", "os", "ly", "<mask>", "Ġ3", "As", "d", "!")
    val expectedIds = Array(3, 4, 5, 6, 7, 8, 2, 10, 11, 12, 13)

    val tokenized = bpeTokenizer.tokenize(sentence)
    val encoded = bpeTokenizer.encode(tokenized)

    for (i <- encoded.indices) {
      assert(encoded(i).wordpiece == expected(i))
      assert(encoded(i).pieceId == expectedIds(i))
    }
  }

  "BpeTokenizer" should "handle empty sentences" taggedAs FastTest in {
    val text = " \n"
    val sentence = Sentence(text, 0, text.length - 1, 0)

    val tokenized = bpeTokenizer.tokenize(sentence)
    val encoded = bpeTokenizer.encode(tokenized)
    assert(tokenized.isEmpty)
    assert(encoded.isEmpty)
  }

  "BpeTokenizer" should "add sentence padding correctly if requested" taggedAs FastTest in {
    val tokenizer = new BpeTokenizer(merges, vocab, padWithSentenceTokens = true)

    val text = "I unambigouosly <mask> 3Asd!"

    val sentence = Sentence(text, 0, text.length - 1, 0)
    val expected = Array("<s>", "I", "Ġunamb", "ig", "ou", "os", "ly", "<mask>", "Ġ3", "As", "d", "!", "</s>")
    val expectedIds = Array(0, 3, 4, 5, 6, 7, 8, 2, 10, 11, 12, 13, 1)

    val tokenized = tokenizer.tokenize(sentence)
    val encoded = tokenizer.encode(tokenized)

    assert(tokenized.head.token == "<s>")
    assert(tokenized.last.token == "</s>")
    for (i <- encoded.indices) {
      assert(encoded(i).wordpiece == expected(i))
      assert(encoded(i).pieceId == expectedIds(i))
    }

  }

  "BpeTokenizer" should "throw exception when a word is not in the vocabulary" taggedAs FastTest in {
    val tokenizer = new BpeTokenizer(merges, vocab, padWithSentenceTokens = true)

    val text = "not in vocabulary"
    val sentence = Sentence(text, 0, text.length - 1, 0)

    val tokenized = tokenizer.tokenize(sentence)
    assertThrows[IllegalArgumentException] {
      tokenizer.encode(tokenized)
    }
  }

  "BpeTokenizer" should "throw exception when an unsupported model type is used" taggedAs FastTest in {
    assertThrows[IllegalArgumentException] {
      new BpeTokenizer(merges, vocab, "deberta")
    }
  }
}
