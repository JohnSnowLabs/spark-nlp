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

import scala.util.Random

/** Regression coverage for `predictSequenceGrouped` (the cross-row batched
  * sequence-classification path added to flatten `*ForSequenceClassification` annotators - see
  * XXXForClassification.scala).
  *
  * Two things must hold:
  *   1. With `coalesceSentences=false`, results must exactly match calling the per-row
  *      [[predictSequence]] once per row and concatenating - regardless of batchSize or how
  *      length-bucketing reorders sentences internally. 2. With `coalesceSentences=true`, each
  *      row must still get exactly ONE annotation, averaged over only that row's own sentences -
  *      not averaged across the whole cross-row batch. This is the part that's easy to get wrong
  *      once a single inference call spans multiple rows: "coalesce the document" has to mean
  *      "coalesce the row", not "coalesce the call".
  */
class XXXForClassificationPredictSequenceGroupedTestSpec extends AnyFlatSpec {

  class FakeSequenceClassification extends XXXForClassification {
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
          TokenPiece(
            wordpiece = tok.token,
            token = tok.token,
            pieceId = tok.begin + 1,
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

    override def tag(batch: Seq[Array[Int]]): Seq[Array[Array[Float]]] =
      throw new NotImplementedError("not exercised by this spec")

    /** Score for a sentence is a pure function of the token ids actually present in its own
      * encoded row (sum of non-special ids, split into two classes) - never a function of which
      * other sentences share its batch or of padding, so batching/bucketing/cross-row mixing can
      * never legitimately change a sentence's own result.
      */
    override def tagSequence(batch: Seq[Array[Int]], activation: String): Array[Array[Float]] =
      batch.map { encoded =>
        val real = encoded.filter(id =>
          id != sentenceStartTokenId && id != sentenceEndTokenId && id != sentencePadTokenId)
        val sum = real.sum.toFloat
        Array(sum, -sum)
      }.toArray

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
        tokenPiece: TokenPiece): Option[IndexedToken] = None
  }

  private val classTags = Map("pos" -> 0, "neg" -> 1)

  /** Builds `numRows` rows, each with 0-3 sentences and each sentence 1-5 tokens, all with
    * globally unique begin/end offsets.
    */
  private def randomRows(
      numRows: Int,
      seed: Long,
      minSentencesPerRow: Int = 0): (Seq[Seq[TokenizedSentence]], Seq[Seq[Sentence]]) = {
    val rnd = new Random(seed)
    var cursor = 0
    val rows = (0 until numRows).map { _ =>
      val numSentences = minSentencesPerRow + rnd.nextInt(4 - minSentencesPerRow)
      (0 until numSentences).map { sentenceIdx =>
        val numTokens = 1 + rnd.nextInt(5)
        val tokens = (0 until numTokens).map { _ =>
          val begin = cursor
          val end = cursor + 2
          cursor += 4
          IndexedToken(s"tok$begin", begin, end)
        }.toArray
        val tokenizedSentence = TokenizedSentence(tokens, sentenceIdx)
        val sentStart = tokens.head.begin
        val sentEnd = tokens.last.end
        val sentence = Sentence(s"s$sentStart", sentStart, sentEnd, sentenceIdx)
        (tokenizedSentence, sentence)
      }
    }
    (rows.map(_.map(_._1)), rows.map(_.map(_._2)))
  }

  private def goldenReference(
      model: FakeSequenceClassification,
      rowsOfTokenizedSentences: Seq[Seq[TokenizedSentence]],
      rowsOfSentences: Seq[Seq[Sentence]],
      coalesceSentences: Boolean): Seq[Seq[com.johnsnowlabs.nlp.Annotation]] =
    rowsOfTokenizedSentences.zip(rowsOfSentences).map { case (tokenizedSentences, sentences) =>
      if (tokenizedSentences.isEmpty) Seq.empty[com.johnsnowlabs.nlp.Annotation]
      else
        model.predictSequence(
          tokenizedSentences,
          sentences,
          batchSize = 1000,
          maxSentenceLength = 512,
          caseSensitive = true,
          coalesceSentences = coalesceSentences,
          classTags,
          ActivationFunction.softmax)
    }

  Seq(1, 2, 3, 5, 8, 16).foreach { batchSize =>
    it should s"match the per-row predictSequence() reference exactly with batchSize=$batchSize (coalesceSentences=false)" taggedAs FastTest in {
      val model = new FakeSequenceClassification
      val (tokenizedRows, sentenceRows) = randomRows(numRows = 12, seed = 99)

      val expected =
        goldenReference(model, tokenizedRows, sentenceRows, coalesceSentences = false)
      val actual = model.predictSequenceGrouped(
        tokenizedRows,
        sentenceRows,
        batchSize = batchSize,
        maxSentenceLength = 512,
        caseSensitive = true,
        coalesceSentences = false,
        classTags,
        ActivationFunction.softmax)

      assert(actual.length == expected.length)
      tokenizedRows.indices.foreach { rowIdx =>
        assert(
          actual(rowIdx) == expected(rowIdx),
          s"row $rowIdx mismatched at batchSize=$batchSize:\n  actual=${actual(
              rowIdx)}\n  expected=${expected(rowIdx)}")
      }
    }
  }

  it should "coalesce PER ROW, not across the whole cross-row batch" taggedAs FastTest in {
    val model = new FakeSequenceClassification
    // every row has at least 2 sentences so a bug that coalesces across the whole call (instead
    // of per row) would visibly average in other rows' scores
    val (tokenizedRows, sentenceRows) =
      randomRows(numRows = 6, seed = 123, minSentencesPerRow = 2)

    val expected = goldenReference(model, tokenizedRows, sentenceRows, coalesceSentences = true)
    val actual = model.predictSequenceGrouped(
      tokenizedRows,
      sentenceRows,
      batchSize = 3, // forces multiple rows' sentences into the same inference batch
      maxSentenceLength = 512,
      caseSensitive = true,
      coalesceSentences = true,
      classTags,
      ActivationFunction.softmax)

    assert(actual.length == 6)
    actual.foreach(rowResult =>
      assert(
        rowResult.length == 1,
        "coalesceSentences must yield exactly one annotation per row"))
    tokenizedRows.indices.foreach { rowIdx =>
      assert(
        actual(rowIdx) == expected(rowIdx),
        s"row $rowIdx mismatched:\n  actual=${actual(rowIdx)}\n  expected=${expected(rowIdx)}")
    }
  }

  it should "leave every row empty when all rows have zero sentences" taggedAs FastTest in {
    val model = new FakeSequenceClassification
    val rows: Seq[Seq[TokenizedSentence]] = Seq(Seq.empty, Seq.empty, Seq.empty)
    val sentenceRows: Seq[Seq[Sentence]] = Seq(Seq.empty, Seq.empty, Seq.empty)

    val result = model.predictSequenceGrouped(
      rows,
      sentenceRows,
      batchSize = 8,
      maxSentenceLength = 512,
      caseSensitive = true,
      coalesceSentences = false,
      classTags,
      ActivationFunction.softmax)

    assert(result.length == 3)
    assert(result.forall(_.isEmpty))
  }

  it should "not leak annotations across rows when empty rows are interleaved with non-empty ones" taggedAs FastTest in {
    val model = new FakeSequenceClassification
    val (tokenizedNonEmpty, sentenceNonEmpty) =
      randomRows(numRows = 4, seed = 55, minSentencesPerRow = 1)

    val tokenizedRows: Seq[Seq[TokenizedSentence]] =
      Seq(
        Seq.empty,
        tokenizedNonEmpty(0),
        Seq.empty,
        Seq.empty,
        tokenizedNonEmpty(1),
        tokenizedNonEmpty(2),
        Seq.empty,
        tokenizedNonEmpty(3))
    val sentenceRows: Seq[Seq[Sentence]] =
      Seq(
        Seq.empty,
        sentenceNonEmpty(0),
        Seq.empty,
        Seq.empty,
        sentenceNonEmpty(1),
        sentenceNonEmpty(2),
        Seq.empty,
        sentenceNonEmpty(3))

    val expectedNonEmpty =
      goldenReference(model, tokenizedNonEmpty, sentenceNonEmpty, coalesceSentences = false)
    val actual = model.predictSequenceGrouped(
      tokenizedRows,
      sentenceRows,
      batchSize = 3,
      maxSentenceLength = 512,
      caseSensitive = true,
      coalesceSentences = false,
      classTags,
      ActivationFunction.softmax)

    assert(actual.length == tokenizedRows.length)
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
