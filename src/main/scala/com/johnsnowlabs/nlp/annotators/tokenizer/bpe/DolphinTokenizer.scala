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

package com.johnsnowlabs.nlp.annotators.tokenizer.bpe

import com.johnsnowlabs.nlp.annotators.common.IndexedToken

import java.io.ByteArrayOutputStream
import java.nio.charset.StandardCharsets
import scala.collection.mutable.ListBuffer
import scala.util.matching.Regex

/** Byte-level BPE tokenizer for ByteDance Dolphin.
  *
  * Unlike [[Gpt2Tokenizer]] it adds no prefix space (Dolphin's ByteLevel has `add_prefix_space:
  * false`), and it decodes as HuggingFace `tokenizers` does: a token is byte-decoded only when
  * every character is in the byte-level alphabet, otherwise its own UTF-8 is emitted, which keeps
  * the 23,944 added tokens (CJK characters, table markup) intact.
  */
private[johnsnowlabs] class DolphinTokenizer(
    merges: Map[(String, String), Int],
    vocab: Map[String, Int],
    specialTokens: SpecialTokens,
    padWithSequenceTokens: Boolean = false,
    addPrefixSpaceToSentence: Boolean = false,
    alwaysAddPrefix: Boolean = false)
    extends BpeTokenizer(
      merges,
      vocab,
      specialTokens,
      padWithSequenceTokens,
      addPrefixSpaceToSentence,
      alwaysAddPrefix) {

  /** GPT-2's byte -> unicode mapping. */
  protected val bytesToUnicodeMapping: Map[Int, String] = {
    val bytes: ListBuffer[Int] =
      ListBuffer.range('!', '~' + 1) ++ ListBuffer.range('¡', '¬' + 1) ++ ListBuffer
        .range('®', 'ÿ' + 1)
    val characters: ListBuffer[Int] = bytes.clone
    var n = 0
    for (b <- 0 until 256) {
      if (!bytes.contains(b)) {
        bytes += b
        characters += (256 + n)
        n += 1
      }
    }
    (bytes zip characters.map(_.toChar.toString)).toMap
  }

  protected val unicodeToByteMapping: Map[String, Int] =
    bytesToUnicodeMapping.map(x => (x._2, x._1))

  protected val decoderVocab: Map[Int, String] = vocab.map(x => (x._2, x._1))

  override val prefixForPieceId: Option[String] = None

  override def preProcessTokenForBpe(token: String): String =
    token
      .getBytes("UTF-8")
      .map { b => if (b < 0) 256 + b else b }
      .foldLeft("")(_ + bytesToUnicodeMapping(_))

  val splitPattern: Regex =
    raw"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""".r

  override def tokenizeSubText(text: String, indexOffset: Int): Array[IndexedToken] =
    splitPattern
      .findAllMatchIn(text)
      .map(tok => IndexedToken(tok.matched, tok.start + indexOffset, tok.end + indexOffset - 1))
      .toArray

  /** Encode a task prompt into `decoder_input_ids`, `<s>{prompt} <Answer/>`. Built directly
    * because [[BpeTokenizer.tokenize]] drops a special token at the start of the text.
    */
  def encodePrompt(prompt: String): Array[Int] = {
    val body = encode(tokenizeSubText(prompt, 0)).map(_.pieceId)
    Array(specialTokens.sentenceStart.id) ++ body ++ Array(vocab(DolphinTokenizer.AnswerToken))
  }

  /** Decode generated ids; ids missing from the vocabulary are skipped. */
  def decodeTokens(tokens: Array[Int]): String = {
    val out = new ByteArrayOutputStream()
    tokens.foreach { id =>
      decoderVocab.get(id).foreach { token =>
        if (token.nonEmpty && token.forall(c => unicodeToByteMapping.contains(c.toString)))
          token.foreach(c => out.write(unicodeToByteMapping(c.toString)))
        else
          out.write(token.getBytes(StandardCharsets.UTF_8))
      }
    }
    // malformed sequences become U+FFFD, matching Rust's String::from_utf8_lossy
    new String(out.toByteArray, StandardCharsets.UTF_8)
  }
}

private[johnsnowlabs] object DolphinTokenizer {

  /** The task prompts. Formula and code prompts exist in Dolphin 1.5; 1.0 reads both as text. */
  val LayoutPrompt: String = "Parse the reading order of this document."
  val TextPrompt: String = "Read text in the image."
  val TablePrompt: String = "Parse the table in the image."
  val FormulaPrompt: String = "Read formula in the image."
  val CodePrompt: String = "Read code in the image."

  /** Separates the task prompt from the generated answer: one token (id 73920) with a leading
    * space.
    */
  val AnswerToken: String = " <Answer/>"

  /** Ids of the wrapped prompts from `transformers` 4.47.1, checked at model load. */
  val GoldenPromptIds: Map[String, Array[Int]] = Map(
    LayoutPrompt -> Array(0, 47928, 286, 7996, 1160, 299, 495, 5262, 36, 73920),
    TextPrompt -> Array(0, 17657, 3360, 301, 286, 1990, 36, 73920),
    TablePrompt -> Array(0, 47928, 286, 4345, 301, 286, 1990, 36, 73920),
    FormulaPrompt -> Array(0, 17657, 3914, 301, 286, 1990, 36, 73920),
    CodePrompt -> Array(0, 17657, 3355, 301, 286, 1990, 36, 73920))

  /** Dolphin has no mask token, so `<unk>` stands in. */
  def specialTokens(vocab: Map[String, Int]): SpecialTokens = SpecialTokens(
    vocab,
    startTokenString = "<s>",
    endTokenString = "</s>",
    unkTokenString = "<unk>",
    maskTokenString = "<unk>",
    padTokenString = "<pad>",
    additionalStrings = Array(AnswerToken))
}
