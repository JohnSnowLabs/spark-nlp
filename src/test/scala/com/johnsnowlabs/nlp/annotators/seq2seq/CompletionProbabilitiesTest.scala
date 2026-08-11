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

import com.johnsnowlabs.tags.FastTest
import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.scalatest.flatspec.AnyFlatSpec

class CompletionProbabilitiesTest extends AnyFlatSpec {

  import CompletionProbabilities.TokenSpan

  /** Builds a `completion_probabilities` array whose tokens spell out `text`, one token per
    * character, so a test can talk about text rather than byte arrays.
    */
  private def probabilitiesFor(text: String, logProb: Double = -0.5): String = {
    val tokens = text.map { char =>
      val bytes = char.toString.getBytes("UTF-8").map(_ & 0xff)
      s"""{"logprob": $logProb, "bytes": [${bytes.mkString(", ")}]}"""
    }
    tokens.mkString("[", ", ", "]")
  }

  private def decodedTexts(json: String): Seq[String] =
    parse(json) match {
      case JArray(tokens) =>
        implicit val formats: Formats = DefaultFormats
        tokens.flatMap { token =>
          (token \ "bytes")
            .extractOpt[Array[Int]]
            .map(bytes => new String(bytes.map(_.toByte), "UTF-8"))
        }
      case _ => Seq.empty
    }

  behavior of "CompletionProbabilities.tokenSpans"

  it should "return contiguous spans over single-byte characters" taggedAs FastTest in {
    val spans = CompletionProbabilities.tokenSpans("""[{"logprob": -0.1, "bytes": [72]},
        | {"logprob": -0.2, "bytes": [105]},
        | {"logprob": -0.3, "bytes": [33]}]""".stripMargin)
    assert(spans == Seq(TokenSpan(0, 1, -0.1), TokenSpan(1, 2, -0.2), TokenSpan(2, 3, -0.3)))
  }

  it should "count a two-byte character as one character" taggedAs FastTest in {
    // "é" (U+00E9) is 0xC3 0xA9 in UTF-8; one token carries both bytes.
    val spans = CompletionProbabilities.tokenSpans(
      """[{"logprob": -0.4, "bytes": [195, 169]}, {"logprob": -0.5, "bytes": [33]}]""")
    assert(spans == Seq(TokenSpan(0, 1, -0.4), TokenSpan(1, 2, -0.5)))
  }

  it should "count a four-byte character as the two UTF-16 units Java strings use" taggedAs FastTest in {
    // U+1F600 GRINNING FACE is 0xF0 0x9F 0x98 0x80, and "😀".length == 2 in Java, so
    // spans must advance by 2 to stay aligned with substring/TokenPiece offsets.
    val spans = CompletionProbabilities.tokenSpans(
      """[{"logprob": -0.1, "bytes": [240, 159, 152, 128]}, {"logprob": -0.2, "bytes": [33]}]""")
    assert(spans == Seq(TokenSpan(0, 2, -0.1), TokenSpan(2, 3, -0.2)))
    assert("😀!".length == 3)
  }

  it should "keep offsets exact when a multi-byte character is split across two tokens" taggedAs FastTest in {
    // llama.cpp's byte-level BPE can split "é" so one token holds 0xC3 and the next 0xA9. The
    // first token completes no character (zero-length span); the second gets the whole one.
    val spans = CompletionProbabilities.tokenSpans("""[{"logprob": -0.4, "bytes": [195]},
        | {"logprob": -0.5, "bytes": [169]},
        | {"logprob": -0.6, "bytes": [33]}]""".stripMargin)
    assert(spans == Seq(TokenSpan(0, 0, -0.4), TokenSpan(0, 1, -0.5), TokenSpan(1, 2, -0.6)))
  }

  it should "produce spans that stay monotonic and end at the text length" taggedAs FastTest in {
    val text = "Le café coûte 5€ — d'accord ?"
    val spans = CompletionProbabilities.tokenSpans(probabilitiesFor(text))
    assert(spans.last.end == text.length)
    assert(spans.map(_.begin) == spans.map(_.begin).sorted)
    assert(spans.forall(s => s.end >= s.begin))
  }

  it should "return empty for malformed, non-array, or incomplete input" taggedAs FastTest in {
    assert(CompletionProbabilities.tokenSpans("not json").isEmpty)
    assert(CompletionProbabilities.tokenSpans("""{"logprob": -0.1}""").isEmpty)
    assert(CompletionProbabilities.tokenSpans("[]").isEmpty)
    // no `bytes` field -> nothing to derive a span from
    assert(CompletionProbabilities.tokenSpans("""[{"logprob": -0.1}]""").isEmpty)
  }

  behavior of "CompletionProbabilities.sliceToCharRange"

  it should "return the input unchanged when the window covers the whole completion" taggedAs FastTest in {
    val json = probabilitiesFor("Paris")
    assert(CompletionProbabilities.sliceToCharRange(json, 0, 5).contains(json))
  }

  it should "drop the tokens of a stripped leading block" taggedAs FastTest in {
    val raw = "<think>hmm</think>Paris"
    val json = probabilitiesFor(raw)
    val kept = CompletionProbabilities.sliceToCharRange(json, raw.indexOf("Paris"), raw.length)
    assert(kept.isDefined)
    assert(decodedTexts(kept.get).mkString == "Paris")
  }

  it should "renumber surviving spans from zero so they match the stripped text" taggedAs FastTest in {
    val raw = "<think>hmm</think>Paris"
    val json = probabilitiesFor(raw)
    val kept =
      CompletionProbabilities.sliceToCharRange(json, raw.indexOf("Paris"), raw.length).get
    val spans = CompletionProbabilities.tokenSpans(kept)
    assert(spans.head.begin == 0)
    assert(spans.last.end == "Paris".length)
  }

  it should "drop tokens of trailing whitespace that trimming removed" taggedAs FastTest in {
    val raw = "Paris   "
    val kept = CompletionProbabilities.sliceToCharRange(probabilitiesFor(raw), 0, 5).get
    assert(decodedTexts(kept).mkString == "Paris")
  }

  it should "keep a token that straddles the window boundary" taggedAs FastTest in {
    // One token covering chars [0,4) with the window starting at 2: it did produce part of the
    // kept text, so dropping it would lose a real logprob.
    val json =
      """[{"logprob": -0.1, "bytes": [97, 98, 99, 100]}, {"logprob": -0.2, "bytes": [101]}]"""
    val kept = CompletionProbabilities.sliceToCharRange(json, 2, 5).get
    assert(decodedTexts(kept) == Seq("abcd", "e"))
  }

  it should "return None when no token falls inside the window" taggedAs FastTest in {
    assert(CompletionProbabilities.sliceToCharRange(probabilitiesFor("Paris"), 10, 20).isEmpty)
  }

  it should "return None for malformed input rather than throwing" taggedAs FastTest in {
    assert(CompletionProbabilities.sliceToCharRange("not json", 0, 5).isEmpty)
    assert(CompletionProbabilities.sliceToCharRange("[]", 0, 5).isEmpty)
  }

  it should "preserve each surviving token object verbatim, including top_logprobs" taggedAs FastTest in {
    val json =
      """[{"logprob": -0.1, "bytes": [97], "top_logprobs": [{"logprob": -0.1, "token": "a"}]},
        | {"logprob": -0.2, "bytes": [98], "top_logprobs": [{"logprob": -0.2, "token": "b"}]}]""".stripMargin
    val kept = CompletionProbabilities.sliceToCharRange(json, 1, 2).get
    parse(kept) match {
      case JArray(tokens) =>
        assert(tokens.length == 1)
        assert((tokens.head \ "top_logprobs") != JNothing)
      case _ => fail("expected a JSON array")
    }
  }
}
