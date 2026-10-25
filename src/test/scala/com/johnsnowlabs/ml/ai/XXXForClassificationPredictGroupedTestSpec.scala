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
import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec

import scala.util.Random

/** Regression coverage for `predictGrouped` (the cross-row batched token-classification path
  * added to flatten `*ForTokenClassification` annotators - see XXXForClassification.scala).
  *
  * `predictGrouped` batches inference across every sentence from every row in one pass, using
  * length-bucketing (sort-by-length, batch, restore order) to limit padding waste. It must:
  *   1. Produce the exact same annotations as calling the original per-row [[predict]] once per
  *      row and concatenating - regardless of `batchSize`, row count, or sentence-length spread
  *      (which drives how length-bucketing reorders things internally). 2. Keep each row's own
  *      annotations in original sentence/token order, since NER-style consumers (e.g.
  *      NerConverter) rely on adjacency for BIO-tag reconstruction. 3. Correctly attribute
  *      annotations to their originating row with no cross-row leakage, and handle empty rows
  *      without the row-misalignment bug fixed elsewhere in this branch (predictGrouped never
  *      filters-then-zipWithIndex; it always allocates a row-indexed array).
  *
  * The fake `tag` stub below returns, for a token's encoded id, a score that depends only on that
  * token's own id - never on neighboring padding or batch composition - so it behaves like a
  * correctly-masked real model for the purpose of this test: any observed difference between
  * `predictGrouped` and the per-row golden reference must come from the grouping/regrouping logic
  * itself, not from padding artifacts.
  */
class XXXForClassificationPredictGroupedTestSpec extends AnyFlatSpec {

  class FakeTokenClassification extends XXXForClassification {
    override protected val sentencePadTokenId: Int = 0
    override protected val sentenceStartTokenId: Int = 101
    override protected val sentenceEndTokenId: Int = 102
    override protected val sigmoidThreshold: Float = 0.5f

    override def tokenizeWithAlignment(
        sentences: Seq[TokenizedSentence],
        maxSeqLength: Int,
        caseSensitive: Boolean): Seq[WordpieceTokenizedSentence] =
      sentences.map { ts =>
        WordpieceTokenizedSentence(ts.indexedTokens.map { tok =>
          // pieceId derived from begin offset: unique per token across the whole test, stable
          // regardless of which row/batch the token ends up grouped into.
          TokenPiece(
            wordpiece = tok.token,
            token = tok.token,
            pieceId = tok.begin + 1, // +1 to stay clear of pad/special ids (0-2)
            isWordStart = true,
            begin = tok.begin,
            end = tok.end)
        })
      }

    override def tokenizeSeqString(
        candidateLabels: Seq[String],
        maxSeqLength: Int,
        caseSensitive: Boolean): Seq[WordpieceTokenizedSentence] = Seq.empty

    override def tokenizeDocument(
        docs: Seq[com.johnsnowlabs.nlp.Annotation],
        maxSeqLength: Int,
        caseSensitive: Boolean): Seq[WordpieceTokenizedSentence] = Seq.empty

    /** Score for a token only depends on its own encoded id - no cross-position mixing, so
      * padding/batch composition can never change a real position's result (mirrors what
      * attention masking guarantees in a correctly implemented real model).
      */
    override def tag(batch: Seq[Array[Int]]): Seq[Array[Array[Float]]] =
      batch.map(_.map(id => Array(id.toFloat, -id.toFloat)))

    override def tagSequence(batch: Seq[Array[Int]], activation: String): Array[Array[Float]] =
      throw new NotImplementedError("not exercised by this spec")

    override def tagZeroShotSequence(
        batch: Seq[Array[Int]],
        entailmentId: Int,
        contradictionId: Int,
        activation: String): Array[Array[Float]] =
      throw new NotImplementedError("not exercised by this spec")

    override def tagSpan(batch: Seq[Array[Int]]): (Array[Array[Float]], Array[Array[Float]]) =
      throw new NotImplementedError("not exercised by this spec")

    override def findIndexedToken(
        tokenizedSentences: Seq[TokenizedSentence],
        sentence: (WordpieceTokenizedSentence, Int),
        tokenPiece: TokenPiece): Option[IndexedToken] =
      tokenizedSentences(sentence._2).indexedTokens.find(p => p.begin == tokenPiece.begin)
  }

  private val classTags = Map("A" -> 0, "B" -> 1)

  /** Builds `numRows` rows, each with a random number of sentences (0-3) and each sentence a
    * random number of tokens (1-6), with globally unique begin/end offsets so tokens are
    * unambiguously identifiable regardless of which row/sentence they end up attributed to.
    */
  private def randomRows(numRows: Int, seed: Long): Seq[Seq[TokenizedSentence]] = {
    val rnd = new Random(seed)
    var cursor = 0
    (0 until numRows).map { _ =>
      val numSentences = rnd.nextInt(4) // 0..3, so some rows are empty
      (0 until numSentences).map { sentenceIdx =>
        val numTokens = 1 + rnd.nextInt(6) // 1..6
        val tokens = (0 until numTokens).map { _ =>
          val begin = cursor
          val end = cursor + 2
          cursor += 4
          IndexedToken(s"tok$begin", begin, end)
        }.toArray
        TokenizedSentence(tokens, sentenceIdx)
      }
    }
  }

  private def goldenReference(
      model: FakeTokenClassification,
      rows: Seq[Seq[TokenizedSentence]]): Seq[Seq[com.johnsnowlabs.nlp.Annotation]] =
    rows.map { rowSentences =>
      // large batchSize so the per-row reference is never itself split across multiple batches
      model.predict(
        rowSentences,
        batchSize = 1000,
        maxSentenceLength = 512,
        caseSensitive = true,
        classTags)
    }

  Seq(1, 2, 3, 5, 8, 16).foreach { batchSize =>
    it should s"match the per-row predict() reference exactly with batchSize=$batchSize" taggedAs FastTest in {
      val model = new FakeTokenClassification
      val rows = randomRows(numRows = 12, seed = 42)

      val expected = goldenReference(model, rows)
      val actual = model.predictGrouped(
        rows,
        batchSize = batchSize,
        maxSentenceLength = 512,
        caseSensitive = true,
        classTags)

      assert(actual.length == expected.length, "one result per row")
      rows.indices.foreach { rowIdx =>
        assert(
          actual(rowIdx) == expected(rowIdx),
          s"row $rowIdx mismatched at batchSize=$batchSize:\n  actual=${actual(
              rowIdx)}\n  expected=${expected(rowIdx)}")
      }
    }
  }

  it should "leave every row empty when all rows have zero sentences" taggedAs FastTest in {
    val model = new FakeTokenClassification
    val rows: Seq[Seq[TokenizedSentence]] = Seq(Seq.empty, Seq.empty, Seq.empty)

    val result = model.predictGrouped(
      rows,
      batchSize = 8,
      maxSentenceLength = 512,
      caseSensitive = true,
      classTags)

    assert(result.length == 3)
    assert(result.forall(_.isEmpty))
  }

  it should "not leak annotations across rows when empty rows are interleaved with non-empty ones" taggedAs FastTest in {
    val model = new FakeTokenClassification
    val nonEmptyRows = randomRows(numRows = 4, seed = 7)
    // interleave: empty, row0, empty, empty, row1, row2, empty, row3
    val rows: Seq[Seq[TokenizedSentence]] =
      Seq(
        Seq.empty,
        nonEmptyRows(0),
        Seq.empty,
        Seq.empty,
        nonEmptyRows(1),
        nonEmptyRows(2),
        Seq.empty,
        nonEmptyRows(3))

    val expectedNonEmpty = goldenReference(model, nonEmptyRows)
    val actual = model.predictGrouped(
      rows,
      batchSize = 3,
      maxSentenceLength = 512,
      caseSensitive = true,
      classTags)

    assert(actual.length == rows.length)
    assert(actual(0).isEmpty)
    assert(actual(2).isEmpty)
    assert(actual(3).isEmpty)
    assert(actual(6).isEmpty)
    assert(actual(1) == expectedNonEmpty(0))
    assert(actual(4) == expectedNonEmpty(1))
    assert(actual(5) == expectedNonEmpty(2))
    assert(actual(7) == expectedNonEmpty(3))
  }
}
