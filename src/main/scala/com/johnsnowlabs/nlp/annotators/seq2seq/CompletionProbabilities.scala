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
package com.johnsnowlabs.nlp.annotators.seq2seq

import org.json4s._
import org.json4s.jackson.JsonMethods._

/** Reading and reshaping the verbatim `completion_probabilities` JSON that `AutoGGUFModel` writes
  * to annotation metadata when `outputLogProbs` is enabled: an array of `{"logprob": ...,
  * "bytes": [...], "top_logprobs": [...], ...}` objects, one per generated token, exactly as
  * llama.cpp returned it.
  *
  * This lives next to `AutoGGUFModel` (rather than next to its main consumer,
  * `LLMUncertaintyEstimator`) because the array has to be kept in sync with the completion text
  * at the point where that text is post-processed - see [[CompletionPostProcessing]].
  */
private[nlp] object CompletionProbabilities {

  private implicit val jsonFormats: Formats = DefaultFormats

  /** One generated token's character span into the completion text, and its log-likelihood.
    *
    * `begin` is inclusive and `end` exclusive, both in Java `String` character (UTF-16 code unit)
    * positions, so they line up directly with `TokenPiece.begin`/`end` and with `substring`.
    */
  case class TokenSpan(begin: Int, end: Int, logProb: Double)

  /** Number of UTF-16 code units a UTF-8 sequence starting with `leadByte` decodes to, and how
    * many bytes long that sequence is. Malformed lead bytes are treated as a single one-unit
    * character (which is what a decoder's replacement character would cost anyway).
    */
  private def sequenceLength(leadByte: Int): Int = {
    val b = leadByte & 0xff
    if (b < 0x80) 1
    else if ((b & 0xe0) == 0xc0) 2
    else if ((b & 0xf0) == 0xe0) 3
    else if ((b & 0xf8) == 0xf0) 4
    else 1
  }

  /** Walks the token array once, tracking each token's character span into the completion text.
    *
    * Spans are accumulated incrementally rather than by decoding the growing byte prefix at every
    * step (which would be quadratic in completion length). llama.cpp's byte-level BPE can split a
    * multi-byte UTF-8 character across two tokens, so an incomplete trailing sequence is carried
    * over into the next token: the token that contains only the first half of a character
    * contributes no characters (a zero-length span), and the token that completes it gets the
    * whole character. Character positions therefore stay exact and monotonic in every case.
    *
    * @return
    *   one entry per token that carries both a numeric `logprob` and an integer-array `bytes`
    *   field, paired with the raw token object so callers can re-emit it verbatim
    */
  private def scan(tokens: Seq[JValue]): Seq[(JValue, TokenSpan)] = {
    var charCount = 0
    var pending = Array.emptyByteArray

    tokens.flatMap { token =>
      for {
        logProb <- (token \ "logprob").extractOpt[Double]
        // Matched as a JArray rather than extracted straight to Array[Int]: json4s turns a
        // missing field into an empty array, which would silently pass a malformed token off as
        // one that generated no characters.
        byteInts <- (token \ "bytes") match {
          case JArray(_) => (token \ "bytes").extractOpt[Array[Int]]
          case _ => None
        }
      } yield {
        val buffer = pending ++ byteInts.map(_.toByte)
        var consumed = 0
        var produced = 0
        var complete = true
        while (complete && consumed < buffer.length) {
          val width = sequenceLength(buffer(consumed).toInt)
          if (consumed + width > buffer.length) complete = false
          else {
            produced += (if (width == 4) 2 else 1) // supplementary chars are a surrogate pair
            consumed += width
          }
        }
        pending = buffer.drop(consumed)

        val begin = charCount
        charCount += produced
        (token, TokenSpan(begin, charCount, logProb))
      }
    }
  }

  /** Per-token character spans and logprobs parsed from the verbatim `completion_probabilities`
    * JSON.
    *
    * @return
    *   empty if `json` does not parse to a non-empty array of objects containing numeric
    *   `logprob` and integer-array `bytes` fields
    */
  def tokenSpans(json: String): Seq[TokenSpan] = {
    try {
      parse(json) match {
        case JArray(tokens) => scan(tokens).map(_._2)
        case _ => Seq.empty
      }
    } catch {
      case _: Exception => Seq.empty
    }
  }

  /** Keeps only the tokens that generated characters `[from, until)` of the completion text,
    * re-emitting each surviving token object verbatim.
    *
    * This is what keeps the array aligned with the annotation's `result` after
    * [[CompletionPostProcessing]] strips a `<think>` block and trims surrounding whitespace:
    * without it, the reasoning block's tokens would still be in the array while the text they
    * produced is gone, silently shifting every character offset a consumer computes from it.
    *
    * @return
    *   `None` if `json` does not parse to an array of well-formed tokens, or if no token survives
    *   the window; the input unchanged when the window already covers the whole completion
    */
  def sliceToCharRange(json: String, from: Int, until: Int): Option[String] = {
    try {
      parse(json) match {
        case JArray(tokens) =>
          val scanned = scan(tokens)
          if (scanned.isEmpty) None
          else if (from <= 0 && until >= scanned.last._2.end) Some(json) // nothing was stripped
          else {
            // Overlap, not containment: a token straddling the boundary still produced part of
            // the kept text, so dropping it would lose a real logprob.
            val kept = scanned.collect {
              case (token, span) if span.end > from && span.begin < until => token
            }
            if (kept.isEmpty) None else Some(compact(render(JArray(kept.toList))))
          }
        case _ => None
      }
    } catch {
      case _: Exception => None
    }
  }
}
