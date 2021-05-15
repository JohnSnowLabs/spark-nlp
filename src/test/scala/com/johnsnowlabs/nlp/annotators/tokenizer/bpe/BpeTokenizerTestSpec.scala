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

import com.johnsnowlabs.nlp.annotators.common.{Sentence, TokenPiece}
import com.johnsnowlabs.tags.FastTest
import org.scalatest.FlatSpec

class BpeTokenizerTestSpec extends FlatSpec {
    val vocab: Map[String, Int] =
      Array(
        "<s>",
        "</s>",
        "<mask>",
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
        "Ġ!",
        "<unk>",
        "<pad>",
//        "ĠI",
//        "ĠAs",
//        "Ġ!",
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
      "Ġ 3",
    ).map(_.split(" ")).map { case Array(c1, c2) => (c1, c2) }.zipWithIndex.toMap

  val bpeTokenizer: BpeTokenizer = BpeTokenizer.forModel(
    "roberta",
    merges,
    vocab,
    padWithSentenceTokens = false
  )

  private def assertEncodedCorrectly(text: String,
                                     encoded: Array[TokenPiece],
                                     expected: Array[String],
                                     expectedIds: Array[Int]): Unit = {
    println(encoded.mkString("Array(\n  ", ",\n  ", "\n)"))
    for (i <- encoded.indices) {
      val piece = encoded(i)
      assert(piece.wordpiece == expected(i))
      assert(piece.pieceId == expectedIds(i))

      assert(text.slice(piece.begin, piece.end + 1) == piece.wordpiece.replace("Ġ", " "))
    }
  }

  "RobertaTokenizer" should "encode words correctly" taggedAs FastTest in {
    val text = "I unambigouosly good 3Asd!"
    val sentence = Sentence(text, 0, text.length - 1, 0)

    val expected: Array[String] = Array("I", "Ġunamb", "ig", "ou", "os", "ly", "Ġgood", "Ġ3", "As", "d", "!")
    val expectedIds: Array[Int] = Array(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)

    val tokenized = bpeTokenizer.tokenize(sentence)
    val encoded = bpeTokenizer.encode(tokenized)

    assertEncodedCorrectly(text, encoded, expected, expectedIds)

  }

  "RobertaTokenizer" should "encode sentences with special tokens correctly" taggedAs FastTest in {
    val text = "I unambigouosly <mask> 3Asd!"
    val sentence = Sentence(text, 0, text.length - 1, 0)

    val expected = Array("I", "Ġunamb", "ig", "ou", "os", "ly", "<mask>", "Ġ3", "As", "d", "!")
    val expectedIds = Array(3, 4, 5, 6, 7, 8, 2, 10, 11, 12, 13)

    val tokenizedWithMask = bpeTokenizer.tokenize(sentence)
    val encoded = bpeTokenizer.encode(tokenizedWithMask)

    assertEncodedCorrectly(text, encoded, expected, expectedIds)
  }

  "RobertaTokenizer" should "handle empty sentences" taggedAs FastTest in {
    val text = " \n"
    val sentence = Sentence(text, 0, text.length - 1, 0)

    val tokenized = bpeTokenizer.tokenize(sentence)
    val encoded = bpeTokenizer.encode(tokenized)
    assert(tokenized.isEmpty)
    assert(encoded.isEmpty)
  }

  "RobertaTokenizer" should "add sentence padding correctly if requested" taggedAs FastTest in {
    val tokenizer = BpeTokenizer.forModel("roberta", merges, vocab, padWithSentenceTokens = true)

    val text = "I unambigouosly <mask> 3Asd!"
    val sentence = Sentence(text, 0, text.length - 1, 0)

    val expected = Array("<s>", "I", "Ġunamb", "ig", "ou", "os", "ly", "<mask>", "Ġ3", "As", "d", "!", "</s>")
    val expectedIds = Array(0, 3, 4, 5, 6, 7, 8, 2, 10, 11, 12, 13, 1)

    val tokenized = tokenizer.tokenize(sentence)
    val encoded = tokenizer.encode(tokenized)

    val textPadded = "<s>I unambigouosly <mask> 3Asd!</s>"
    assertEncodedCorrectly(textPadded, encoded, expected, expectedIds)

    assert(tokenized.head.token == "<s>")
    assert(tokenized.last.token == "</s>")

  }
  "RobertaTokenizer" should "handle unknown words" taggedAs FastTest in {
    val text = "???"
    val sentence = Sentence(text, 0, text.length - 1, 0)

    val tokenized = bpeTokenizer.tokenize(sentence)
    val encoded = bpeTokenizer.encode(tokenized)
    assert(encoded.forall(_.pieceId == vocab("<unk>")))
  }
  "RobertaTokenizer" should "throw exception when an unsupported model type is used" taggedAs FastTest in {
    assertThrows[IllegalArgumentException] {
      BpeTokenizer.forModel("unsupported", merges, vocab, padWithSentenceTokens = false)
    }
  }

//  "RobertaTokenizer" should "encode 2" taggedAs FastTest in {
//    val text = "Rare Hendrix song draft sells for almost $17,000"
////    val sentence = Sentence(text, 0, text.length - 1, 0)
//    val indexedTokens = text.split(" ").map(
//      tok => IndexedToken(tok, text.indexOf(tok), text.indexOf(tok) + tok.length - 1)
//    )
//    println(indexedTokens.mkString("Array(\n  ", ",\n  ", "\n)"))
//
//    val indexedTokSentences: Array[IndexedToken] = indexedTokens
//      .map(tok => Sentence(tok.token, tok.begin, tok.begin + tok.token.length - 1, 0))
//      .flatMap(bpeTokenizer.tokenize)
//    println(indexedTokSentences.mkString("Array(\n  ", ",\n  ", "\n)"))
//
//    val encoded = bpeTokenizer.encode(indexedTokSentences)
//    println(encoded.mkString("Array(\n  ", ",\n  ", "\n)"))
//    for (i <- encoded.indices) {
//      val piece = encoded(i)
//      println("asserting ", text.slice(piece.begin, piece.end + 1), piece.wordpiece.replace("Ġ", " "))
//      assert(text.slice(piece.begin, piece.end + 1) == piece.wordpiece.replace("Ġ", " "))
//
//    }
//  }
}
