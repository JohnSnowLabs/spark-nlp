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

import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec

import scala.util.Random

/** Regression coverage for `predictSpanGrouped` (the new batched question-answering path added to
  * flatten `*ForQuestionAnswering` annotators). `predictSpan` itself had no `batchSize` at all -
  * it always ran exactly one example per `tagSpan` call - so this is new code, not a refactor of
  * an existing batched path, and it has to get three things right that `predictSpan` got away
  * with only because its batch was always size 1:
  *
  *   1. `startLogits.transpose.map(_.sum / startLogits.length)` in predictSpan looks like a
  *      batch-dimension unwrap but is actually an AVERAGE. With batch size 1 that average is a
  *      no-op; with more than one example sharing a batch it would blend different questions'
  *      answer scores together. predictSpanGrouped must index per example instead. 2. tagSpan's
  *      ONNX path needs a rectangular batch (same length every row); predictSpanGrouped must pad
  *      shorter examples in a batch up to that batch's own max length. 3. Padding must use
  *      `sentencePadTokenId`, not a hardcoded 0 - this fake deliberately uses a non-zero pad id
  *      (999) to catch any such hardcoding.
  *
  * The fake `tagSpan` below scores every position as a triangular peak centered on a position
  * that is a pure function of that example's OWN real (non-padding) tokens - never a function of
  * neighboring examples or how much trailing padding was added - so any observed difference
  * between predictSpanGrouped and the per-row predictSpan reference must come from the
  * batching/regrouping logic itself.
  */
class XXXForClassificationPredictSpanGroupedTestSpec extends AnyFlatSpec {

  class FakeSpanClassification extends XXXForClassification {
    override protected val sentencePadTokenId: Int = 999 // deliberately non-zero
    override protected val sentenceStartTokenId: Int = 101
    override protected val sentenceEndTokenId: Int = 102
    override protected val sigmoidThreshold: Float = 0.5f

    override def tokenizeWithAlignment(
        sentences: Seq[TokenizedSentence],
        maxSeqLength: Int,
        caseSensitive: Boolean): Seq[WordpieceTokenizedSentence] = Seq.empty

    override def tokenizeSeqString(
        candidateLabels: Seq[String],
        maxSeqLength: Int,
        caseSensitive: Boolean): Seq[WordpieceTokenizedSentence] = Seq.empty

    /** One WordpieceTokenizedSentence per input document/annotation, with a single token whose
      * pieceId is derived from the annotation's own begin offset - unique per document across the
      * whole test.
      */
    override def tokenizeDocument(
        docs: Seq[Annotation],
        maxSeqLength: Int,
        caseSensitive: Boolean): Seq[WordpieceTokenizedSentence] =
      docs.map { doc =>
        WordpieceTokenizedSentence(
          Array(
            TokenPiece(
              wordpiece = doc.result,
              token = doc.result,
              pieceId = doc.begin + 1,
              isWordStart = true,
              begin = doc.begin,
              end = doc.end)))
      }

    override def tag(batch: Seq[Array[Int]]): Seq[Array[Array[Float]]] =
      throw new NotImplementedError("not exercised by this spec")

    override def tagSequence(batch: Seq[Array[Int]], activation: String): Array[Array[Float]] =
      throw new NotImplementedError("not exercised by this spec")

    override def tagZeroShotSequence(
        batch: Seq[Array[Int]],
        entailmentId: Int,
        contradictionId: Int,
        activation: String): Array[Array[Float]] =
      throw new NotImplementedError("not exercised by this spec")

    /** Peaks (score 0, decaying by 1 per position away) exactly at a target position derived from
      * that row's own real content - start target = position right after the question's own end
      * token (first occurrence of sentenceEndTokenId); end target = the last non-padding
      * position. Neither depends on other rows sharing the batch or on how much padding was
      * added, so the argmax is stable regardless of batching/bucketing.
      */
    override def tagSpan(batch: Seq[Array[Int]]): (Array[Array[Float]], Array[Array[Float]]) = {
      val maxLen = batch.map(_.length).max
      // offset by +1000 so the peak score is unambiguously positive: -math.abs(0) is -0.0f, and
      // predictSpan's transpose-average (`Array(-0.0f).sum`) incidentally normalizes that to
      // +0.0f while predictSpanGrouped's direct indexing does not - same numeric value either way
      // (-0.0f == 0.0f), but the toString stored in annotation metadata would differ. Avoid the
      // edge case entirely rather than mask it.
      def peakAt(target: Int): Array[Float] =
        Array.tabulate(maxLen)(i => 1000f - math.abs(i - target).toFloat)

      val starts = batch.map { row =>
        val target = row.indexOf(sentenceEndTokenId)
        peakAt(target)
      }.toArray
      val ends = batch.map { row =>
        val target = row.lastIndexWhere(_ != sentencePadTokenId)
        peakAt(target)
      }.toArray
      (starts, ends)
    }

    override def findIndexedToken(
        tokenizedSentences: Seq[TokenizedSentence],
        sentence: (WordpieceTokenizedSentence, Int),
        tokenPiece: TokenPiece): Option[IndexedToken] = None
  }

  /** Builds `numRows` rows, each either empty or a [question, context] pair of DOCUMENT
    * annotations with globally unique begin/end offsets, so every token is unambiguously
    * identifiable.
    */
  private def randomRows(
      numRows: Int,
      seed: Long,
      allEmpty: Boolean = false): Seq[Seq[Annotation]] = {
    val rnd = new Random(seed)
    var cursor = 0
    (0 until numRows).map { _ =>
      if (allEmpty || rnd.nextInt(5) == 0) Seq.empty
      else {
        def nextDoc(): Annotation = {
          val begin = cursor
          val len = 1 + rnd.nextInt(6)
          val end = begin + len
          cursor += len + 2
          Annotation(AnnotatorType.DOCUMENT, begin, end, s"doc$begin", Map.empty)
        }
        Seq(nextDoc(), nextDoc()) // [question, context]
      }
    }
  }

  private def goldenReference(
      model: FakeSpanClassification,
      rows: Seq[Seq[Annotation]]): Seq[Seq[Annotation]] =
    rows.map { documents =>
      if (documents.isEmpty) Seq.empty[Annotation]
      else model.predictSpan(documents, maxSentenceLength = 512, caseSensitive = true)
    }

  Seq(1, 2, 3, 5, 8).foreach { batchSize =>
    it should s"match the per-row predictSpan() reference exactly with batchSize=$batchSize" taggedAs FastTest in {
      val model = new FakeSpanClassification
      val rows = randomRows(numRows = 15, seed = 42)

      val expected = goldenReference(model, rows)
      val actual =
        model.predictSpanGrouped(
          rows,
          batchSize = batchSize,
          maxSentenceLength = 512,
          caseSensitive = true)

      assert(actual.length == expected.length)
      rows.indices.foreach { rowIdx =>
        assert(
          actual(rowIdx) == expected(rowIdx),
          s"row $rowIdx mismatched at batchSize=$batchSize:\n  actual=${actual(
              rowIdx)}\n  expected=${expected(rowIdx)}")
      }
    }
  }

  it should "pad shorter examples up to the batch's own max length without changing their answer" taggedAs FastTest in {
    val model = new FakeSpanClassification
    val rnd = new Random(7)
    var cursor = 0
    // deliberately mix a very short row with much longer ones in the same batch
    def doc(len: Int): Annotation = {
      val begin = cursor
      val end = begin + len
      cursor += len + 2
      Annotation(AnnotatorType.DOCUMENT, begin, end, s"doc$begin", Map.empty)
    }
    val shortRow = Seq(doc(1), doc(1))
    val longRow = Seq(doc(20), doc(20))
    val rows = Seq(shortRow, longRow, shortRow, longRow)

    val expected = goldenReference(model, rows)
    val actual =
      model.predictSpanGrouped(rows, batchSize = 4, maxSentenceLength = 512, caseSensitive = true)

    assert(actual == expected)
  }

  it should "leave every row empty when all rows have no documents" taggedAs FastTest in {
    val model = new FakeSpanClassification
    val rows = randomRows(numRows = 4, seed = 1, allEmpty = true)

    val result =
      model.predictSpanGrouped(rows, batchSize = 8, maxSentenceLength = 512, caseSensitive = true)

    assert(result.length == 4)
    assert(result.forall(_.isEmpty))
  }

  it should "not leak answers across rows when empty rows are interleaved with non-empty ones" taggedAs FastTest in {
    val model = new FakeSpanClassification
    val nonEmpty = randomRows(numRows = 4, seed = 2).filter(_.nonEmpty)
    assert(nonEmpty.length >= 3, "need at least 3 non-empty rows for this test to be meaningful")

    val rows: Seq[Seq[Annotation]] =
      Seq(Seq.empty, nonEmpty(0), Seq.empty, Seq.empty, nonEmpty(1), nonEmpty(2), Seq.empty)

    val expectedNonEmpty = goldenReference(model, nonEmpty)
    val actual =
      model.predictSpanGrouped(rows, batchSize = 2, maxSentenceLength = 512, caseSensitive = true)

    assert(actual.length == rows.length)
    assert(actual(0).isEmpty)
    assert(actual(2).isEmpty)
    assert(actual(3).isEmpty)
    assert(actual(6).isEmpty)
    assert(actual(1) == expectedNonEmpty(0))
    assert(actual(4) == expectedNonEmpty(1))
    assert(actual(5) == expectedNonEmpty(2))
  }
}
