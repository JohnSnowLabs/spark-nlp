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
      .strip(
        Array(
          "<think>ok</think>Paris",
          "<think>still going, got cut off",
          "no tag at all"))
    assert(result sameElements Array("Paris", "", "no tag at all"))
  }

  it should "only match the configured tag name" taggedAs FastTest in {
    val result = harness
      .setRemoveThinkingTag("think")
      .strip(Array("<reasoning>not the configured tag</reasoning>Paris"))
    assert(result sameElements Array("<reasoning>not the configured tag</reasoning>Paris"))
  }
}
