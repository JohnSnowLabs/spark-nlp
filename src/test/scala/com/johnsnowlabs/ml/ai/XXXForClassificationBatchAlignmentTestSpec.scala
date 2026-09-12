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

package com.johnsnowlabs.ml.ai

import com.johnsnowlabs.nlp.ActivationFunction
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec

/** Regression coverage for the logit/sentence pairing bug in `predictSequence` and
  * `predictSequenceWithZeroShot` (XXXForClassification.scala).
  *
  * Prior to the fix, aggregation happened inside the `.grouped(batchSize).flatMap { batch => ...
  * }` loop against the FULL outer `sentences` parameter instead of the current batch's own slice.
  * `sentences.zip(logits)` then paired batch 2+'s scores with sentences 0..batchSize-1 (zip
  * truncates to the shorter side), so any row with more sentences than `batchSize` got some
  * annotations anchored at the wrong sentence and carrying another sentence's classification
  * scores. Row *counts* stayed correct, which is why this was silent.
  *
  * These tests exercise `predictSequence`/`predictSequenceWithZeroShot` directly against a
  * minimal fake implementation of the trait (no TensorFlow/ONNX session, no downloaded model, no
  * Spark session needed) so they run fast and deterministically. Each fake sentence's identity is
  * threaded through tokenization -> encoding -> tagging via its wordpiece `pieceId`, so the
  * stubbed `tagSequence`/`tagZeroShotSequence` can produce a score that is a pure function of
  * which sentence it was actually given, making any cross-sentence mispairing observable.
  *
  * NOTE: this file does not cover the separate row-misalignment fix (`.filter(_.nonEmpty)` before
  * `.zipWithIndex` in 13 annotators' `batchAnnotate`, e.g. MarianTransformer, T5Transformer, and
  * the embeddings that batch across rows). That fix was verified by direct diff review since
  * exercising it end-to-end requires a loaded model via `.pretrained()` / `.loadSavedModel()`,
  * which needs network access or local test-only model resources not available in all
  * environments. Once local test model resources are available, add empty-row-in-the-middle
  * DataFrame tests to MarianTestSpec / T5TestSpec / E5EmbeddingsTestSpec etc. asserting that a
  * non-empty row's output is unaffected by an empty row appearing before it.
  */
class XXXForClassificationBatchAlignmentTestSpec extends AnyFlatSpec {

  /** Fake classification head. Sentence identity is smuggled through as the wordpiece pieceId
    * (offset by ID_OFFSET to stay clear of the special token ids), so the stubbed tag methods can
    * recover "which sentence is this really" from the encoded token ids they were handed.
    */
  class FakeClassification extends XXXForClassification {
    override protected val sentencePadTokenId: Int = 0
    override protected val sentenceStartTokenId: Int = 101
    override protected val sentenceEndTokenId: Int = 102
    override protected val sigmoidThreshold: Float = 0.5f

    val ID_OFFSET = 1000

    override def tokenizeWithAlignment(
        sentences: Seq[TokenizedSentence],
        maxSeqLength: Int,
        caseSensitive: Boolean): Seq[WordpieceTokenizedSentence] =
      sentences.map { ts =>
        WordpieceTokenizedSentence(
          Array(TokenPiece(
            wordpiece = ts.tokens.headOption.getOrElse(""),
            token = ts.tokens.headOption.getOrElse(""),
            pieceId = ts.sentenceIndex + ID_OFFSET,
            isWordStart = true,
            begin = 0,
            end = 0)))
      }

    override def tokenizeSeqString(
        candidateLabels: Seq[String],
        maxSeqLength: Int,
        caseSensitive: Boolean): Seq[WordpieceTokenizedSentence] =
      candidateLabels.zipWithIndex.map { case (label, i) =>
        WordpieceTokenizedSentence(
          Array(TokenPiece(label, label, pieceId = i, isWordStart = true, begin = 0, end = 0)))
      }

    override def tokenizeDocument(
        docs: Seq[com.johnsnowlabs.nlp.Annotation],
        maxSeqLength: Int,
        caseSensitive: Boolean): Seq[WordpieceTokenizedSentence] = Seq.empty

    override def tag(batch: Seq[Array[Int]]): Seq[Array[Array[Float]]] =
      batch.map(_ => Array(Array(0f)))

    override def tagSpan(batch: Seq[Array[Int]]): (Array[Array[Float]], Array[Array[Float]]) =
      (Array(Array(0f)), Array(Array(0f)))

    override def findIndexedToken(
        tokenizedSentences: Seq[TokenizedSentence],
        sentence: (WordpieceTokenizedSentence, Int),
        tokenPiece: TokenPiece): Option[IndexedToken] = None

    /** Decodes each row's true sentence identity from its own encoded token ids (position 1,
      * right after the start token) and returns a 2-class score that strongly favors class 0 for
      * even identities and class 1 for odd identities. If a row is ever paired with the wrong
      * sentence's position, its returned label/metadata will disagree with its own identity's
      * parity.
      */
    override def tagSequence(batch: Seq[Array[Int]], activation: String): Array[Array[Float]] =
      batch.map { encoded =>
        val identity = encoded(1) - ID_OFFSET
        if (identity % 2 == 0) Array(10f, 0f) else Array(0f, 10f)
      }.toArray

    override def tagZeroShotSequence(
        batch: Seq[Array[Int]],
        entailmentId: Int,
        contradictionId: Int,
        activation: String): Array[Array[Float]] =
      batch.map { encoded =>
        // encodeSequence layout: [start, questionToken, qEnd, labelToken, end].
        // position 1 = this sentence's identity (from tokenizeWithAlignment's pieceId);
        // position 3 = which candidate label this row is scoring (0="even", 1="odd", from
        // tokenizeSeqString's pieceId). Score entailment high only when the label matches the
        // sentence's own parity, so softmax over labels actually discriminates.
        val identity = encoded(1) - ID_OFFSET
        val labelId = encoded(3)
        val correctLabelId = if (identity % 2 == 0) 0 else 1
        if (labelId == correctLabelId) Array(10f, 0f) else Array(0f, 10f)
      }.toArray
  }

  private def sentenceAt(i: Int): (TokenizedSentence, Sentence) = {
    val content = s"s$i"
    val tokenized = TokenizedSentence(Array(IndexedToken(content, 0, content.length - 1)), i)
    val sentence = Sentence(content, start = i * 10, end = i * 10 + content.length - 1, index = i)
    (tokenized, sentence)
  }

  private def buildInput(n: Int): (Seq[TokenizedSentence], Seq[Sentence]) = {
    val pairs = (0 until n).map(sentenceAt)
    (pairs.map(_._1), pairs.map(_._2))
  }

  "predictSequence" should "anchor every annotation at its own originating sentence across multiple batches" taggedAs FastTest in {
    val model = new FakeClassification
    val (tokenized, sentences) = buildInput(5)

    val result = model.predictSequence(
      tokenized,
      sentences,
      batchSize = 2, // forces 3 batches: [0,1], [2,3], [4]
      maxSentenceLength = 128,
      caseSensitive = true,
      coalesceSentences = false,
      tags = Map("even" -> 0, "odd" -> 1),
      activation = ActivationFunction.softmax)

    assert(result.length == 5, "one annotation per sentence")

    val sentenceIndices = result.map(_.metadata("sentence").toInt).sorted
    assert(
      sentenceIndices == (0 until 5),
      s"every original sentence index must appear exactly once, got $sentenceIndices")

    result.foreach { annotation =>
      val idx = annotation.metadata("sentence").toInt
      val expectedLabel = if (idx % 2 == 0) "even" else "odd"
      assert(
        annotation.result == expectedLabel,
        s"sentence $idx got label ${annotation.result}, expected $expectedLabel " +
          "(a mismatch here means the annotation was scored using a different sentence's batch)")
    }
  }

  it should "average scores over ALL sentences, not just the last batch, when coalesceSentences=true" taggedAs FastTest in {
    val model = new FakeClassification
    val (tokenized, sentences) = buildInput(5) // identities 0,1,2,3,4 -> 3 even, 2 odd

    val result = model.predictSequence(
      tokenized,
      sentences,
      batchSize = 2,
      maxSentenceLength = 128,
      caseSensitive = true,
      coalesceSentences = true,
      tags = Map("even" -> 0, "odd" -> 1),
      activation = ActivationFunction.softmax)

    assert(
      result.length == 1,
      "coalesceSentences must produce exactly one annotation, not one per batch")

    val evenScore = result.head.metadata("even").toFloat
    val oddScore = result.head.metadata("odd").toFloat
    // 3 identities score (10,0), 2 identities score (0,10) -> average = (30/5, 20/5) = (6, 4)
    assert(
      math.abs(evenScore - 6f) < 1e-3f,
      s"expected averaged 'even' score ~6.0, got $evenScore")
    assert(math.abs(oddScore - 4f) < 1e-3f, s"expected averaged 'odd' score ~4.0, got $oddScore")
  }

  it should "keep per-sentence label sets correctly attributed under sigmoid activation across batches" taggedAs FastTest in {
    val model = new FakeClassification
    val (tokenized, sentences) = buildInput(5)

    val result = model.predictSequence(
      tokenized,
      sentences,
      batchSize = 2,
      maxSentenceLength = 128,
      caseSensitive = true,
      coalesceSentences = false,
      tags = Map("even" -> 0, "odd" -> 1),
      activation = ActivationFunction.sigmoid)

    val bySentence = result.groupBy(_.metadata("sentence").toInt)
    (0 until 5).foreach { idx =>
      val labels = bySentence.getOrElse(idx, Seq.empty).map(_.result).toSet
      val expected = if (idx % 2 == 0) Set("even") else Set("odd")
      assert(labels == expected, s"sentence $idx got labels $labels, expected $expected")
    }
  }

  "predictSequenceWithZeroShot" should "anchor every annotation at its own originating sentence across multiple batches" taggedAs FastTest in {
    val model = new FakeClassification
    val (tokenized, sentences) = buildInput(4)

    val result = model.predictSequenceWithZeroShot(
      tokenized,
      sentences,
      candidateLabels = Array("even", "odd"),
      entailmentId = 0,
      contradictionId = 1,
      batchSize = 1, // forces one sentence per batch: worst case for the old bug
      maxSentenceLength = 128,
      caseSensitive = true,
      coalesceSentences = false,
      tags = Map.empty,
      activation = ActivationFunction.softmax)

    assert(result.length == 4, "one annotation per sentence")

    val sentenceIndices = result.map(_.metadata("sentence").toInt).sorted
    assert(
      sentenceIndices == (0 until 4),
      s"every original sentence index must appear exactly once, got $sentenceIndices " +
        "(duplicates/missing indices indicate later batches were mis-anchored)")

    result.foreach { annotation =>
      val idx = annotation.metadata("sentence").toInt
      val expectedLabel = if (idx % 2 == 0) "even" else "odd"
      assert(
        annotation.result == expectedLabel,
        s"sentence $idx got label ${annotation.result}, expected $expectedLabel")
    }
  }

  it should "coalesce to exactly one annotation averaged across all sentences" taggedAs FastTest in {
    val model = new FakeClassification
    val (tokenized, sentences) = buildInput(4) // 2 even, 2 odd

    val result = model.predictSequenceWithZeroShot(
      tokenized,
      sentences,
      candidateLabels = Array("even", "odd"),
      entailmentId = 0,
      contradictionId = 1,
      batchSize = 1,
      maxSentenceLength = 128,
      caseSensitive = true,
      coalesceSentences = true,
      tags = Map.empty,
      activation = ActivationFunction.softmax)

    assert(result.length == 1, "coalesceSentences must produce exactly one annotation")
  }

  "predictSequence" should "return an empty sequence for empty input without throwing" taggedAs FastTest in {
    val model = new FakeClassification
    val result = model.predictSequence(
      Seq.empty,
      Seq.empty,
      batchSize = 8,
      maxSentenceLength = 128,
      caseSensitive = true,
      coalesceSentences = true, // would previously reach `sentences.head` if not guarded
      tags = Map("even" -> 0, "odd" -> 1),
      activation = ActivationFunction.softmax)
    assert(result.isEmpty)
  }
}
