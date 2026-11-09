/*
 * Copyright 2017-2026 John Snow Labs
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

package com.johnsnowlabs.nlp.annotators.tokenizer

import com.johnsnowlabs.nlp.annotators.common.Sentence
import com.johnsnowlabs.nlp.annotators.tokenizer.bpe.{
  BpeTokenizer,
  DolphinTokenizer,
  SpecialTokens
}
import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec

class DolphinTokenizerTestSpec extends AnyFlatSpec {

  // Dolphin's real vocabulary is 73,921 entries (50,000 byte-level BPE pieces + 23,944 added
  // tokens). This fixture is a miniature with the same *shapes*, because that is what the decode
  // rule keys off: a token is byte-decoded only if every character is in the byte-level alphabet.
  private val G = "Ġ" // the byte-level stand-in for a leading space

  private val vocab: Map[String, Int] = Map(
    // specials
    "<s>" -> 0,
    "<pad>" -> 1,
    "</s>" -> 2,
    "<unk>" -> 3,
    // ordinary byte-level BPE pieces
    "P" -> 10,
    "a" -> 11,
    "r" -> 12,
    "s" -> 13,
    "e" -> 14,
    "Pa" -> 15,
    "Par" -> 16,
    "Pars" -> 17,
    "Parse" -> 18,
    s"${G}the" -> 19,
    // ASCII added tokens: every char IS in the byte-level alphabet, so they byte-decode to
    // themselves. Dolphin emits tables using exactly these.
    "<table>" -> 50001,
    "<tr>" -> 50002,
    "<td>" -> 50003,
    "</td>" -> 50004,
    "</tr>" -> 50005,
    "</table>" -> 50006,
    "[TMP_52]" -> 50007,
    // CJK added tokens: NOT in the byte-level alphabet. Gpt2Tokenizer.decodeTokens throws here.
    "吐" -> 60001, // 吐
    "착" -> 60002, // 착
    // U+0144: the first code point past the byte-level alphabet's 0..255 range
    "ń" -> 71300,
    // an added token containing a space: ' ' is not in the byte-level alphabet either
    " <Answer/>" -> 73920)

  private val merges: Map[(String, String), Int] =
    Map(("P", "a") -> 0, ("Pa", "r") -> 1, ("Par", "s") -> 2, ("Pars", "e") -> 3)

  private val tokenizer: DolphinTokenizer = BpeTokenizer
    .forModel(
      "dolphin",
      merges,
      vocab,
      specialTokens = Some(DolphinTokenizer.specialTokens(vocab)))
    .asInstanceOf[DolphinTokenizer]

  "DolphinTokenizer" should "decode ordinary byte-level BPE pieces" taggedAs FastTest in {
    assert(tokenizer.decodeTokens(Array(18, 19)) == "Parse the")
  }

  it should "decode ASCII added tokens verbatim (table markup must survive)" taggedAs FastTest in {
    val ids = Array(50001, 50002, 50003, 18, 50004, 50005, 50006)
    assert(tokenizer.decodeTokens(ids) == "<table><tr><td>Parse</td></tr></table>")
  }

  // Gpt2Tokenizer.decodeTokens throws NoSuchElementException here, which would break every
  // Chinese document -- one of Dolphin's headline benchmarks.
  it should "decode CJK added tokens instead of throwing" taggedAs FastTest in {
    assert(tokenizer.decodeTokens(Array(60001, 60002)) == "吐착")
  }

  it should "decode a mix of BPE pieces and added tokens" taggedAs FastTest in {
    assert(tokenizer.decodeTokens(Array(18, 19, 60001, 50003, 50007)) == "Parse the吐<td>[TMP_52]")
  }

  it should "decode the added token U+0144 as itself, not as a byte" taggedAs FastTest in {
    assert(tokenizer.decodeTokens(Array(18, 71300)) == "Parseń")
  }

  it should "decode an added token containing a space" taggedAs FastTest in {
    assert(tokenizer.decodeTokens(Array(73920)) == " <Answer/>")
  }

  it should "skip ids that are not in the vocabulary rather than throwing" taggedAs FastTest in {
    assert(tokenizer.decodeTokens(Array(18, 999999, 19)) == "Parse the")
  }

  // Dolphin's ByteLevel sets add_prefix_space=false. Gpt2Tokenizer prepends a space when no
  // prependString is set, which would turn `Parse` (47928) into `ĠParse` (36360) in the real
  // vocabulary and silently change the first token of every task prompt.
  it should "not prepend a space when encoding" taggedAs FastTest in {
    val pieces = tokenizer.encode(tokenizer.tokenize(Sentence("Parse", 0, 4, 0)))
    assert(pieces.map(_.pieceId) sameElements Array(18))
    assert(!pieces.exists(_.wordpiece.startsWith(G)))
  }

  it should "encode a task prompt as <s> + body + ' <Answer/>'" taggedAs FastTest in {
    assert(tokenizer.encodePrompt("Parse") sameElements Array(0, 18, 73920))
  }

  it should "always start the prompt with <s> and end it with the answer token" taggedAs FastTest in {
    Seq("Parse", "Parse the", "").foreach { body =>
      val ids = tokenizer.encodePrompt(body)
      assert(ids.head == 0, s"prompt '$body' lost its leading <s>")
      assert(ids.last == 73920, s"prompt '$body' lost its trailing answer token")
    }
  }

  it should "expose the five task prompts and their golden ids" taggedAs FastTest in {
    assert(DolphinTokenizer.GoldenPromptIds.size == 5)
    assert(
      DolphinTokenizer.GoldenPromptIds(DolphinTokenizer.LayoutPrompt) sameElements
        Array(0, 47928, 286, 7996, 1160, 299, 495, 5262, 36, 73920))
    // every golden sequence starts with <s> and ends with " <Answer/>"
    DolphinTokenizer.GoldenPromptIds.values.foreach { ids =>
      assert(ids.head == 0)
      assert(ids.last == 73920)
    }
  }

  it should "register a dolphin case in SpecialTokens.getSpecialTokensForModel" taggedAs FastTest in {
    val st: SpecialTokens = SpecialTokens.getSpecialTokensForModel("dolphin", vocab)
    assert(st.sentenceStart.id == 0)
    assert(st.sentenceEnd.id == 2)
    assert(st.pad.id == 1)
    assert(st.contains(" <Answer/>"))
  }
}
