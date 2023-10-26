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

import com.johnsnowlabs.nlp.annotators.common.Sentence
import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec

class Gpt2TokenizerTestSpec extends AnyFlatSpec with BpeTokenizerBehaviours {
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
      "As",
      "d",
      "!").zipWithIndex.toMap

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
    "Ġ I").map(_.split(" ")).map { case Array(c1, c2) => (c1, c2) }.zipWithIndex.toMap

  val modelType: String = "gpt2"

  override val replaceCharBeforeAssertion: Some[String] = Some("Ġ")

  "Gpt2Tokenizer" should behave like correctBpeTokenizer(
    text = " I unambigouosly good 3Asd!",
    expected = Array("ĠI", "Ġunamb", "ig", "ou", "os", "ly", "Ġgood", "Ġ3", "As", "d", "!"),
    expectedIds = Array(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11))

  it should behave like correctBpeTokenizerInFringeSituations()

  it should behave like correctBpeTokenizerSpecialTokens(
    text = " I unambigouosly <|endoftext|> good 3Asd <|endoftext|>",
    expected = Array(
      "ĠI",
      "Ġunamb",
      "ig",
      "ou",
      "os",
      "ly",
      "<|endoftext|>",
      "Ġgood",
      "Ġ3",
      "As",
      "d",
      "<|endoftext|>"),
    expectedIds = Array(1, 2, 3, 4, 5, 6, 0, 7, 8, 9, 10, 0))

  it should "encode non latin tokens" taggedAs FastTest in {
    val text = "吳天恩"

    val vocab: Map[String, Int] =
      "ĠåĲ³å¤©æģ©".map(_.toString).zipWithIndex.toMap ++ Seq(("<|endoftext|>", 100))

    val merges: Map[(String, String), Int] = Map.empty

    val bpeTokenizer =
      BpeTokenizer.forModel(modelType, merges, vocab, alwaysAddPrefix = false)

    val indexedTokens =
      bpeTokenizer.tokenize(Sentence(text, 0, text.length, 0))

    val encodedTokens = bpeTokenizer.encode(indexedTokens)

    assert(
      encodedTokens.forall(_.pieceId != bpeTokenizer.specialTokens.unk.id),
      "Tokens should be able to be encoded.")

  }
}
