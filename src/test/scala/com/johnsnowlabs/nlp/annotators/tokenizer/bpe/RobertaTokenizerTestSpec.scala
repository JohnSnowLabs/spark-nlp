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

import org.scalatest.flatspec.AnyFlatSpec

class RobertaTokenizerTestSpec extends AnyFlatSpec with BpeTokenizerBehaviours {
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
      "I").zipWithIndex.toMap

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
    "Ġ 3").map(_.split(" ")).map { case Array(c1, c2) => (c1, c2) }.zipWithIndex.toMap

  override val modelType = "roberta"

  override val replaceCharBeforeAssertion: Some[String] = Some("Ġ")

  "RobertaTokenizer" should behave like correctBpeTokenizer(
    text = "I unambigouosly good 3Asd!",
    Array("I", "Ġunamb", "ig", "ou", "os", "ly", "Ġgood", "Ġ3", "As", "d", "!"),
    Array(16, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13))

  "RobertaTokenizer" should behave like correctBpeTokenizerWithAddedPrefixSpace(
    text = "I unambigouosly good 3Asd!",
    Array("I", "Ġunamb", "ig", "ou", "os", "ly", "Ġgood", "Ġ3", "As", "d", "!"),
    Array(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13))

  it should behave like correctBpeTokenizerInFringeSituations()

  it should behave like correctBpeTokenizerSpecialTokens(
    text = "I unambigouosly <mask> good 3Asd <mask>",
    expected =
      Array("I", "Ġunamb", "ig", "ou", "os", "ly", "<mask>", "Ġgood", "Ġ3", "As", "d", "<mask>"),
    expectedIds = Array(16, 4, 5, 6, 7, 8, 2, 9, 10, 11, 12, 2))
}
