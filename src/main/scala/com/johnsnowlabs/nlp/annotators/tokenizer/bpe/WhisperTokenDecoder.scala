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

import java.nio.charset.Charset
import scala.collection.mutable.ArrayBuffer

/** Class used by Whisper model to decode tokens. Does not require merges and is therefore
  * omitted.
  *
  * Note that this means this class cannot tokenize strings.
  *
  * @param vocab
  *   Vocabulary of Tokens for decoding
  * @param specialTokens
  *   Special tokens that should be filtered during decoding
  */
class WhisperTokenDecoder(vocab: Map[String, Int], specialTokens: SpecialTokens)
    extends Gpt2Tokenizer(Map.empty, vocab, specialTokens) {

  /** Whisper timestamp tokens are the fixed-form `<|X.XX|>` entries added to the vocabulary (0.00
    * to 30.00 in 0.02s steps, per the official Whisper tokenizer), distinct from the
    * language/task/`<|notimestamps|>` special tokens. Precomputed once so decoding a batch does
    * not re-parse token strings per call.
    */
  private val timestampTokenSeconds: Map[Int, Double] = vocab.collect {
    case (token, id) if token.matches("<\\|\\d+\\.\\d\\d\\|>") =>
      id -> token.stripPrefix("<|").stripSuffix("|>").toDouble
  }

  /** Decodes the token ids into strings.
    *
    * Token IDs, that are not in the vocabulary are decoded to an empty string (some token IDs
    * might not be in the vocabulary).
    *
    * @param tokens
    *   Array of token IDs to decode
    * @return
    *   Decoded string
    */
  override def decodeTokens(tokens: Array[Int]): String = {
    val text = tokens
      .map(token => decoderVocab.getOrElse(token, ""))
      .filter(x => !specialTokens.contains(x))
      .mkString("")
    val bytes = text.map(x => unicodeToByteMapping(x.toString)).map(x => x.toByte).toArray
    new String(bytes, Charset.forName("UTF-8"))
  }

  /** Decodes a generation that was produced with timestamp tokens enabled (i.e.
    * `<|notimestamps|>` was not forced) into a sequence of (startSeconds, endSeconds, text)
    * segments.
    *
    * Whisper emits timestamp tokens in pairs around each segment: a start-timestamp token,
    * followed by that segment's text tokens, followed by an end-timestamp token — repeated for
    * every segment detected within the audio window. This mirrors the reference decoding
    * convention (see `decode_with_timestamps` in the original Whisper implementation): every
    * timestamp token toggles between "opening" and "closing" a segment, and the text between two
    * consecutive timestamp tokens becomes that segment's `result`.
    *
    * Any tokens seen before an opening timestamp, or after generation ends without a closing
    * timestamp, are dropped — a well-formed generation should not produce either, since the first
    * forced token pins the first timestamp and generation stops at `eosTokenId`.
    *
    * @param tokens
    *   Array of token IDs to decode
    * @return
    *   Segments as (startSeconds, endSeconds, text) tuples, in generation order
    */
  def decodeTokensWithTimestamps(tokens: Array[Int]): Seq[(Double, Double, String)] = {
    val segments = new ArrayBuffer[(Double, Double, String)]()
    val currentTextTokens = new ArrayBuffer[Int]()
    var segmentStart: Option[Double] = None

    tokens.foreach { token =>
      timestampTokenSeconds.get(token) match {
        case Some(seconds) =>
          segmentStart match {
            case None =>
              // Opening timestamp for a new segment.
              segmentStart = Some(seconds)
            case Some(start) =>
              // Closing timestamp: emit the segment spanning [start, seconds).
              val text = decodeTokens(currentTextTokens.toArray)
              segments += ((start, seconds, text))
              currentTextTokens.clear()
              segmentStart = None
          }
        case None =>
          if (segmentStart.isDefined) currentTextTokens += token
      }
    }

    segments.toSeq
  }
}
