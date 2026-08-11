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
package com.johnsnowlabs.ml.ai

import com.johnsnowlabs.nlp.annotators.common.TokenPiece
import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec

/** Covers the MARS phrase-grouping algorithm against hand-built logits, with no ONNX model
  * involved - the arithmetic here (which wordpiece starts a phrase, which importance a phrase
  * takes, where its character span ends) is what `LLMUncertaintyEstimator`'s `mars` method
  * ultimately joins against per-token logprobs.
  */
class MarsClassificationTest extends AnyFlatSpec {

  /** A wordpiece covering `[begin, end]` (inclusive, as the wordpiece tokenizer emits). The `##`
    * continuation marker is part of the wordpiece string but not of the underlying text, so it
    * does not count towards the span.
    */
  private def piece(wordpiece: String, begin: Int, isWordStart: Boolean): TokenPiece = {
    val textLength = wordpiece.stripPrefix("##").length
    TokenPiece(wordpiece, wordpiece, 0, isWordStart, begin, begin + textLength - 1)
  }

  /** Logits for one wordpiece: `boundary` picks the argmax over `[0:2]`, `importanceLogit` is the
    * raw pre-sigmoid value at index 2.
    */
  private def logits(boundary: Int, importanceLogit: Float): Array[Float] =
    if (boundary == 0) Array(1.0f, 0.0f, importanceLogit) else Array(0.0f, 1.0f, importanceLogit)

  private def sigmoid(x: Double): Float = (1.0 / (1.0 + math.exp(-x))).toFloat

  behavior of "MarsClassification.groupPhrases"

  it should "return no phrases for an empty answer" taggedAs FastTest in {
    assert(MarsClassification.groupPhrases(Array.empty, Array.empty).isEmpty)
  }

  it should "make each word-starting boundary token its own phrase" taggedAs FastTest in {
    // "Paris is nice" - every token starts a word and is classified as a boundary (class 0)
    val pieces = Array(piece("Paris", 0, true), piece("is", 6, true), piece("nice", 9, true))
    val rows = Array(logits(0, 2.0f), logits(0, -2.0f), logits(0, 0.0f))
    val phrases = MarsClassification.groupPhrases(pieces, rows)
    assert(phrases.map(p => (p.begin, p.end)) sameElements Array((0, 4), (6, 7), (9, 12)))
    assert(phrases.head.importance == sigmoid(2.0))
  }

  it should "keep a wordpiece continuation inside its word's phrase" taggedAs FastTest in {
    // "Kyrgyzstan" tokenized as "kyrgyz" + "##stan": the continuation must never start a phrase,
    // even when the model classifies it as a boundary.
    val pieces = Array(piece("kyrgyz", 0, true), piece("##stan", 6, false), piece("is", 11, true))
    val rows = Array(logits(0, 3.0f), logits(0, -3.0f), logits(0, 0.0f))
    val phrases = MarsClassification.groupPhrases(pieces, rows)
    assert(phrases.map(p => (p.begin, p.end)) sameElements Array((0, 9), (11, 12)))
  }

  it should "extend a phrase across word-starting tokens classified as non-boundary" taggedAs FastTest in {
    // Class 1 means "no break here", so "New" + "York" group into a single phrase.
    val pieces = Array(piece("New", 0, true), piece("York", 4, true), piece("is", 9, true))
    val rows = Array(logits(0, 2.0f), logits(1, 1.0f), logits(0, 0.0f))
    val phrases = MarsClassification.groupPhrases(pieces, rows)
    assert(phrases.map(p => (p.begin, p.end)) sameElements Array((0, 7), (9, 10)))
  }

  it should "take each phrase's importance from its first token" taggedAs FastTest in {
    val pieces = Array(piece("New", 0, true), piece("York", 4, true))
    val rows = Array(logits(0, 4.0f), logits(1, -4.0f))
    val phrases = MarsClassification.groupPhrases(pieces, rows)
    assert(phrases.length == 1)
    assert(phrases.head.importance == sigmoid(4.0))
  }

  it should "emit spans with inclusive ends that cover every answer character" taggedAs FastTest in {
    val answer = "Paris is the capital"
    val pieces = Array(
      piece("Paris", 0, true),
      piece("is", 6, true),
      piece("the", 9, true),
      piece("capital", 13, true))
    val rows = Array.fill(4)(logits(0, 0.0f))
    val phrases = MarsClassification.groupPhrases(pieces, rows)
    assert(phrases.last.end == answer.length - 1)
    phrases.foreach(p => assert(p.end >= p.begin && p.end < answer.length))
  }

  it should "produce exactly one phrase when nothing is a boundary" taggedAs FastTest in {
    val pieces = Array(piece("a", 0, true), piece("b", 2, true), piece("c", 4, true))
    val rows = Array(logits(0, 1.0f), logits(1, 1.0f), logits(1, 1.0f))
    val phrases = MarsClassification.groupPhrases(pieces, rows)
    assert(phrases.length == 1)
    assert(phrases.head.begin == 0 && phrases.head.end == 4)
  }

  it should "reject a logits array that does not line up with the wordpieces" taggedAs FastTest in {
    val pieces = Array(piece("a", 0, true), piece("b", 2, true))
    assertThrows[IllegalArgumentException] {
      MarsClassification.groupPhrases(pieces, Array(logits(0, 0.0f)))
    }
  }
}
