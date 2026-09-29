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

/** Byte-level BPE tokenizer for ByteDance Dolphin (document image parsing).
  *
  * Dolphin's vocabulary is unusual in two ways that both matter here:
  *
  *   1. It is 73,921 entries: 50,000 byte-level BPE pieces plus '''23,944 added tokens''' with
  *      ids >= 50,000. The added tokens are raw CJK characters, `[TMP_n]` markers, and the HTML
  *      table structure tokens (`<table>`, `<tr>`, `<td>`, ...). Every one of them carries
  *      `special: true` in `tokenizer.json`. 2. Its ByteLevel pre-tokenizer sets
  *      `add_prefix_space: false`, unlike GPT-2/RoBERTa.
  *
  * Consequences, and why this class exists rather than reusing [[Gpt2Tokenizer]]:
  *
  *   - `Gpt2Tokenizer.decodeTokens` maps every character through `unicodeToByteMapping`, which
  *     throws `NoSuchElementException` on any character outside the byte-level alphabet — i.e. on
  *     every CJK added token, so on every Chinese document. It also filters out special tokens,
  *     which for Dolphin would delete all `<table>`/`<td>` markup from a parsed table.
  *   - `Gpt2Tokenizer.tokenizeSubText` prepends a space when no `prependString` is set. For
  *     Dolphin that silently changes the first prompt token (`Parse` = 47928 becomes `ĠParse` =
  *     36360).
  *
  * Decoding follows HuggingFace `tokenizers` exactly (`pre_tokenizers/byte_level.rs`): for each
  * token, if '''every''' character is in the byte-level alphabet, byte-decode it; otherwise emit
  * the token's own UTF-8 bytes. That single rule handles all four cases correctly — BPE pieces
  * (`Ġthe`), ASCII added tokens (`<td>`, `[TMP_52]`), CJK added tokens (`吐`), and added tokens
  * containing a space (` <Answer/>`, whose leading space is not in the alphabet, so it is emitted
  * verbatim — which is what HuggingFace does for added tokens anyway).
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

  /** GPT-2's byte -> unicode mapping. Verified to produce ids identical to Dolphin's own
    * five-stage pre-tokenizer chain (NFKC, literal split, individual digits, punctuation
    * isolation, newline isolation, ByteLevel) for the three task prompts; see
    * [[DolphinTokenizer.GoldenPromptIds]].
    */
  protected val bytesToUnicodeMapping: Map[Int, String] = {
    val bytes: ListBuffer[Int] =
      ListBuffer.range('!', '~' + 1) ++ ListBuffer.range('¡', '¬' + 1) ++ ListBuffer
        .range('®', 'ÿ' + 1)
    val characters: ListBuffer[Int] = bytes.clone
    var n = 0
    for (b <- 0 to 256) {
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

  /** Dolphin's ByteLevel has `add_prefix_space: false`, so unlike GPT-2 there is no prefix. */
  override val prefixForPieceId: Option[String] = None

  /** Every Dolphin prompt begins with `<s>`, which the default split path drops. Opting in leaves
    * every other model's tokenization untouched — see
    * [[BpeTokenizer.preserveLeadingSpecialToken]].
    */
  override protected val preserveLeadingSpecialToken: Boolean = true

  override def preProcessTokenForBpe(token: String): String =
    token
      .getBytes("UTF-8")
      .map { b => if (b < 0) 256 + b else b }
      .foldLeft("")(_ + bytesToUnicodeMapping(_))

  val splitPattern: Regex =
    raw"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""".r

  /** Unlike [[Gpt2Tokenizer]] this must NOT prepend a space — see the class doc. */
  override def tokenizeSubText(text: String, indexOffset: Int): Array[IndexedToken] =
    splitPattern
      .findAllMatchIn(text)
      .map(tok => IndexedToken(tok.matched, tok.start + indexOffset, tok.end + indexOffset - 1))
      .toArray

  /** Encode a task prompt into `decoder_input_ids`, i.e. `<s>{prompt} <Answer/>`.
    *
    * Built by construction rather than by running the wrapped string through
    * [[BpeTokenizer.tokenize]], because that path drops a special token that sits at the very
    * start of the text: [[BpeTokenizer.splitOnSpecialToken]] splits with
    * `StringUtils.splitByWholeSeparator`, which discards the leading empty segment, so its `if (i
    * \== 0 && subTextProcessed.isEmpty) result += tok` guard never fires. Every Dolphin prompt
    * starts with `<s>`, so it would be silently lost.
    *
    * Building the template explicitly is also simply more honest: the wrapper is fixed, so there
    * is nothing to discover by re-parsing it.
    */
  def encodePrompt(prompt: String): Array[Int] = {
    val body = encode(tokenizeSubText(prompt, 0)).map(_.pieceId)
    Array(specialTokens.sentenceStart.id) ++ body ++ Array(vocab(DolphinTokenizer.AnswerToken))
  }

  /** Encode arbitrary text with no prompt wrapper. Used only for `customPrompt`. */
  def encodeText(text: String): Array[Int] =
    encode(tokenizeSubText(text, 0)).map(_.pieceId)

  /** Decode with the HuggingFace ByteLevel fallback. Unknown ids are skipped rather than
    * throwing; callers are expected to have already stripped the prompt and truncated at EOS.
    */
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

  /** Strip the decoder prompt prefix, stop at the first EOS, drop padding, then decode. */
  def decodeGenerated(
      tokens: Array[Int],
      promptLength: Int,
      eosTokenId: Int,
      padTokenId: Int): String = {
    val generated = tokens.drop(promptLength)
    val stop = generated.indexOf(eosTokenId) match {
      case -1 => generated.length
      case i => i
    }
    decodeTokens(generated.take(stop).filter(_ != padTokenId)).trim
  }
}

private[johnsnowlabs] object DolphinTokenizer {

  /** The task prompts. Formula and code prompts exist in Dolphin 1.5; 1.0 reads both as text. */
  val LayoutPrompt: String = "Parse the reading order of this document."
  val TextPrompt: String = "Read text in the image."
  val TablePrompt: String = "Parse the table in the image."
  val FormulaPrompt: String = "Read formula in the image."
  val CodePrompt: String = "Read code in the image."

  /** Separates the task prompt from the generated answer. A single token (id 73920) whose content
    * carries a leading space.
    */
  val AnswerToken: String = " <Answer/>"

  /** Dolphin wraps every prompt as `<s>{prompt} <Answer/>` and feeds it as `decoder_input_ids`.
    */
  def wrapPrompt(prompt: String): String = s"<s>$prompt$AnswerToken"

  /** Ids for the wrapped prompts, computed from `tokenizer.json` and verified against
    * `transformers` 4.47.1. Asserted at model load so a tokenizer regression fails loudly instead
    * of silently degrading output quality.
    */
  val GoldenPromptIds: Map[String, Array[Int]] = Map(
    LayoutPrompt -> Array(0, 47928, 286, 7996, 1160, 299, 495, 5262, 36, 73920),
    TextPrompt -> Array(0, 17657, 3360, 301, 286, 1990, 36, 73920),
    TablePrompt -> Array(0, 47928, 286, 4345, 301, 286, 1990, 36, 73920),
    FormulaPrompt -> Array(0, 17657, 3914, 301, 286, 1990, 36, 73920),
    CodePrompt -> Array(0, 17657, 3355, 301, 286, 1990, 36, 73920))

  /** Dolphin has no mask token; `<unk>` stands in. Only these five are registered as special:
    * [[BpeTokenizer.tokenize]] scans `specialTokens.allTokens` on every call, so registering all
    * 23,944 added tokens would make encoding quadratic for no benefit — we only ever encode a
    * closed set of short prompts.
    */
  def specialTokens(vocab: Map[String, Int]): SpecialTokens = SpecialTokens(
    vocab,
    startTokenString = "<s>",
    endTokenString = "</s>",
    unkTokenString = "<unk>",
    maskTokenString = "<unk>",
    padTokenString = "<pad>",
    additionalStrings = Array(AnswerToken))
}
