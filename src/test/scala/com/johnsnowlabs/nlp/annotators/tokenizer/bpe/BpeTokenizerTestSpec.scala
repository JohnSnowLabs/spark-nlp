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
  val bpeTokenizer: BpeTokenizer = BpeTokenizer.forModel("roberta", merges, vocab)

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
    val tokenizer = BpeTokenizer.forModel("roberta", merges, vocab, padWithSentenceTokens = true)

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
  "BpeTokenizer" should "handle unknown words" taggedAs FastTest in {
    val text = "???"
    val sentence = Sentence(text, 0, text.length - 1, 0)

    val tokenized = bpeTokenizer.tokenize(sentence)
    val encoded = bpeTokenizer.encode(tokenized)

    println(tokenized.mkString("Array(\n  ", ",\n  ", "\n)"))
    println(encoded.mkString("Array(\n  ", ",\n  ", "\n)"))

    assert(encoded.forall(_.pieceId == 3))
  }
  "BpeTokenizer" should "throw exception when an unsupported model type is used" taggedAs FastTest in {
    assertThrows[IllegalArgumentException] {
      BpeTokenizer.forModel("unsupported", merges, vocab)
    }
  }
}
