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
import org.apache.spark.ml.param.{ParamMap, Params}
import org.scalatest.flatspec.AnyFlatSpec

class CompletionPostProcessingTest extends AnyFlatSpec {

  private class Harness(override val uid: String) extends Params with CompletionPostProcessing {
    override def copy(extra: ParamMap): Params = defaultCopy(extra)
    def strip(results: Array[String]): Array[String] = processCompletions(results)
    def stripWithOffsets(results: Array[String]): Array[ProcessedCompletion] =
      processCompletionsWithOffsets(results)
  }

  private def harness: Harness = new Harness("completion-post-processing-test")

  behavior of "CompletionPostProcessing.processCompletions"

  it should "leave completions unchanged when no thinking tag is set" taggedAs FastTest in {
    val result = harness.strip(Array("plain answer, no tags"))
    assert(result sameElements Array("plain answer, no tags"))
  }

  it should "strip a closed <think>...</think> block" taggedAs FastTest in {
    val result = harness
      .setRemoveThinkingTag("think")
      .strip(Array("<think>reasoning here</think>The answer is Paris."))
    assert(result sameElements Array("The answer is Paris."))
  }

  it should "strip an unclosed <think> block left by a truncated generation" taggedAs FastTest in {
    // nPredict can cut generation off before the closing tag appears; the raw in-progress
    // reasoning must not leak through unstripped.
    val result = harness
      .setRemoveThinkingTag("think")
      .strip(Array("<think>reasoning that never finished because nPredict cut it off"))
    assert(result sameElements Array(""))
  }

  it should "leave text with no thinking tag present untouched, even with the param set" taggedAs FastTest in {
    val result = harness
      .setRemoveThinkingTag("think")
      .strip(Array("Just a normal answer, no tags at all."))
    assert(result sameElements Array("Just a normal answer, no tags at all."))
  }

  it should "handle a batch mixing closed and unclosed tags independently" taggedAs FastTest in {
    val result = harness
      .setRemoveThinkingTag("think")
      .strip(Array("<think>ok</think>Paris", "<think>still going, got cut off", "no tag at all"))
    assert(result sameElements Array("Paris", "", "no tag at all"))
  }

  it should "only match the configured tag name" taggedAs FastTest in {
    val result = harness
      .setRemoveThinkingTag("think")
      .strip(Array("<reasoning>not the configured tag</reasoning>Paris"))
    assert(result sameElements Array("<reasoning>not the configured tag</reasoning>Paris"))
  }

  behavior of "CompletionPostProcessing offsets"

  // These offsets are what keeps `completion_probabilities` aligned with the text that ends up in
  // the annotation - see AutoGGUFModel.runCompletions.

  it should "report offset 0 when no thinking tag is configured" taggedAs FastTest in {
    val processed = harness.stripWithOffsets(Array("  untouched, not even trimmed  "))
    assert(processed.head.text == "  untouched, not even trimmed  ")
    assert(processed.head.beginOffset.contains(0))
  }

  it should "report where the answer starts after a stripped leading block" taggedAs FastTest in {
    val raw = "<think>reasoning here</think>The answer is Paris."
    val processed = harness.setRemoveThinkingTag("think").stripWithOffsets(Array(raw))
    assert(processed.head.text == "The answer is Paris.")
    assert(processed.head.beginOffset.contains(raw.indexOf("The answer")))
  }

  it should "account for whitespace trimmed between the block and the answer" taggedAs FastTest in {
    val raw = "<think>hmm</think>\n\n  Paris  "
    val processed = harness.setRemoveThinkingTag("think").stripWithOffsets(Array(raw))
    assert(processed.head.text == "Paris")
    assert(processed.head.beginOffset.contains(raw.indexOf("Paris")))
  }

  it should "report an offset that slices the raw completion back to the cleaned text" taggedAs FastTest in {
    // The property AutoGGUFModel relies on: text == raw.substring(offset, offset + text.length)
    val raws = Array(
      "<think>a</think>Paris",
      "  <think>a</think>  Paris  ",
      "no tag here",
      "<think>truncated and never closed")
    val processed = harness.setRemoveThinkingTag("think").stripWithOffsets(raws)
    processed.zip(raws).foreach { case (result, raw) =>
      val offset = result.beginOffset.getOrElse(fail(s"expected a contiguous slice for '$raw'"))
      assert(raw.substring(offset, offset + result.text.length) == result.text)
    }
  }

  it should "report no offset when the block leaves two disjoint pieces of text" taggedAs FastTest in {
    // Nothing sensible to return here: the surviving text is not a contiguous slice of the raw
    // completion, so no single shift maps per-token offsets onto it.
    val processed = harness
      .setRemoveThinkingTag("think")
      .stripWithOffsets(Array("Paris<think>wait, is it?</think> is the capital."))
    assert(processed.head.text == "Paris is the capital.")
    assert(processed.head.beginOffset.isEmpty)
  }

  it should "keep processCompletions byte-identical to the offset-aware path" taggedAs FastTest in {
    val raws = Array("<think>a</think>Paris", "plain", "<think>unclosed", "  spaced  ")
    val configured = harness.setRemoveThinkingTag("think")
    assert(configured.strip(raws) sameElements configured.stripWithOffsets(raws).map(_.text))
  }

  behavior of "thinking-tag removal combined with completion_probabilities"

  /** One token per character of `text`, as llama.cpp would report them. */
  private def probabilitiesFor(text: String): String =
    text
      .map { char =>
        val bytes = char.toString.getBytes("UTF-8").map(_ & 0xff)
        s"""{"logprob": -0.5, "bytes": [${bytes.mkString(", ")}]}"""
      }
      .mkString("[", ", ", "]")

  it should "leave logprob spans indexing the answer, not the stripped reasoning" taggedAs FastTest in {
    // The whole point of tracking offsets: MarsTokenImportance computes its phrase spans against
    // the *stripped* answer, while completion_probabilities covers the *raw* generation. Before
    // the two were reconciled, every span a consumer derived here was shifted by the length of
    // the reasoning block, and MARS silently joined importance onto the wrong tokens.
    val raw = "<think>Let me think about this.</think>\n\nParis"
    val answer = "Paris"

    val processed = harness.setRemoveThinkingTag("think").stripWithOffsets(Array(raw)).head
    assert(processed.text == answer)

    val begin = processed.beginOffset.get
    val aligned = CompletionProbabilities
      .sliceToCharRange(probabilitiesFor(raw), begin, begin + processed.text.length)
      .get
    val spans = CompletionProbabilities.tokenSpans(aligned)

    assert(spans.length == answer.length, "one surviving token per answer character")
    assert(spans.head.begin == 0, "spans must be relative to the answer, not the raw completion")
    assert(spans.last.end == answer.length)

    // A MARS phrase covering the whole answer (inclusive end, as MarsClassification emits) must
    // now pick up every token.
    val phrases = Seq((0, answer.length - 1, 0.9))
    val importance = com.johnsnowlabs.nlp.annotators.uncertainty.UncertaintyMetrics
      .alignByCharSpan(spans.map(s => (s.begin, s.end, -s.logProb)), phrases)
      .map(_._2)
    assert(importance.forall(_ == 0.9), s"every answer token should be weighted, got $importance")
  }
}
