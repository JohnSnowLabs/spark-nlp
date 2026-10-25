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

/** Regression coverage for `predictSequenceWithZeroShotGrouped` (the cross-row batched zero-shot
  * path added to flatten `*ForZeroShotClassification` annotators).
  *
  * Same two properties as the sequence-classification grouped path:
  *   1. `coalesceSentences=false` results must exactly match the per-row
  *      [[predictSequenceWithZeroShot]] reference, regardless of batchSize/bucketing. 2.
  *      `coalesceSentences=true` must average within a row only, never across rows sharing an
  *      inference batch.
  */
class XXXForClassificationPredictZeroShotGroupedTestSpec extends AnyFlatSpec {

  class FakeZeroShotClassification extends XXXForClassification {
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
        caseSensitive: Boolean): Seq[WordpieceTokenizedSentence] =
      candidateLabels.zipWithIndex.map { case (label, i) =>
        WordpieceTokenizedSentence(
          Array(
            TokenPiece(label, label, pieceId = 5000 + i, isWordStart = true, begin = 0, end = 0)))
      }

    override def tokenizeDocument(
        docs: Seq[com.johnsnowlabs.nlp.Annotation],
        maxSeqLength: Int,
        caseSensitive: Boolean): Seq[WordpieceTokenizedSentence] = Seq.empty

    override def tag(batch: Seq[Array[Int]]): Seq[Array[Array[Float]]] =
      throw new NotImplementedError("not exercised by this spec")

    override def tagSequence(batch: Seq[Array[Int]], activation: String): Array[Array[Float]] =
      throw new NotImplementedError("not exercised by this spec")

    /** entailment logit is a pure function of (sentence's own token ids, this row's label id) -
      * never a function of neighboring sentences/padding/batch composition.
      */
    override def tagZeroShotSequence(
        batch: Seq[Array[Int]],
        entailmentId: Int,
        contradictionId: Int,
        activation: String): Array[Array[Float]] =
      batch.map { encoded =>
        val sentenceSum = encoded
          .filter(id => id != sentenceStartTokenId && id != sentenceEndTokenId && id < 5000)
          .sum
        val labelId = encoded.find(_ >= 5000).getOrElse(5000) - 5000
        val score = (sentenceSum + labelId).toFloat
        val row = new Array[Float](math.max(entailmentId, contradictionId) + 1)
        row(entailmentId) = score
        row(contradictionId) = -score
        row
      }.toArray

    override def tagSpan(batch: Seq[Array[Int]]): (Array[Array[Float]], Array[Array[Float]]) =
      throw new NotImplementedError("not exercised by this spec")

    override def findIndexedToken(
        tokenizedSentences: Seq[TokenizedSentence],
        sentence: (WordpieceTokenizedSentence, Int),
        tokenPiece: TokenPiece): Option[IndexedToken] = None
  }

  private def randomRows(
      numRows: Int,
      seed: Long,
      minSentencesPerRow: Int = 0): (Seq[Seq[TokenizedSentence]], Seq[Seq[Sentence]]) = {
    val rnd = new Random(seed)
    var cursor = 0
    val rows = (0 until numRows).map { _ =>
      val numSentences = minSentencesPerRow + rnd.nextInt(4 - minSentencesPerRow)
      (0 until numSentences).map { sentenceIdx =>
        val numTokens = 1 + rnd.nextInt(4)
        val tokens = (0 until numTokens).map { _ =>
          val begin = cursor
          val end = cursor + 2
          cursor += 4
          IndexedToken(s"tok$begin", begin, end)
        }.toArray
        val tokenizedSentence = TokenizedSentence(tokens, sentenceIdx)
        val sentence =
          Sentence(s"s${tokens.head.begin}", tokens.head.begin, tokens.last.end, sentenceIdx)
        (tokenizedSentence, sentence)
      }
    }
    (rows.map(_.map(_._1)), rows.map(_.map(_._2)))
  }

  private val candidateLabels = Array("pos", "neg", "neutral")

  private def goldenReference(
      model: FakeZeroShotClassification,
      rowsOfTokenizedSentences: Seq[Seq[TokenizedSentence]],
      rowsOfSentences: Seq[Seq[Sentence]],
      coalesceSentences: Boolean): Seq[Seq[com.johnsnowlabs.nlp.Annotation]] =
    rowsOfTokenizedSentences.zip(rowsOfSentences).map { case (tokenizedSentences, sentences) =>
      if (tokenizedSentences.isEmpty) Seq.empty[com.johnsnowlabs.nlp.Annotation]
      else
        model.predictSequenceWithZeroShot(
          tokenizedSentences,
          sentences,
          candidateLabels,
          entailmentId = 0,
          contradictionId = 1,
          batchSize = 1000,
          maxSentenceLength = 512,
          caseSensitive = true,
          coalesceSentences = coalesceSentences,
          tags = Map.empty,
          ActivationFunction.softmax)
    }

  Seq(1, 2, 3, 5, 8).foreach { batchSize =>
    it should s"match the per-row predictSequenceWithZeroShot() reference exactly with batchSize=$batchSize" taggedAs FastTest in {
      val model = new FakeZeroShotClassification
      val (tokenizedRows, sentenceRows) = randomRows(numRows = 10, seed = 321)

      val expected =
        goldenReference(model, tokenizedRows, sentenceRows, coalesceSentences = false)
      val actual = model.predictSequenceWithZeroShotGrouped(
        tokenizedRows,
        sentenceRows,
        candidateLabels,
        entailmentId = 0,
        contradictionId = 1,
        batchSize = batchSize,
        maxSentenceLength = 512,
        caseSensitive = true,
        coalesceSentences = false,
        tags = Map.empty,
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
    val model = new FakeZeroShotClassification
    val (tokenizedRows, sentenceRows) =
      randomRows(numRows = 5, seed = 654, minSentencesPerRow = 2)

    val expected = goldenReference(model, tokenizedRows, sentenceRows, coalesceSentences = true)
    val actual = model.predictSequenceWithZeroShotGrouped(
      tokenizedRows,
      sentenceRows,
      candidateLabels,
      entailmentId = 0,
      contradictionId = 1,
      batchSize = 4,
      maxSentenceLength = 512,
      caseSensitive = true,
      coalesceSentences = true,
      tags = Map.empty,
      ActivationFunction.softmax)

    assert(actual.length == 5)
    actual.foreach(rowResult =>
      assert(
        rowResult.length == 1,
        "coalesceSentences must yield exactly one annotation per row"))
    tokenizedRows.indices.foreach { rowIdx =>
      assert(actual(rowIdx) == expected(rowIdx), s"row $rowIdx mismatched")
    }
  }

  it should "leave every row empty when all rows have zero sentences" taggedAs FastTest in {
    val model = new FakeZeroShotClassification
    val rows: Seq[Seq[TokenizedSentence]] = Seq(Seq.empty, Seq.empty)
    val sentenceRows: Seq[Seq[Sentence]] = Seq(Seq.empty, Seq.empty)

    val result = model.predictSequenceWithZeroShotGrouped(
      rows,
      sentenceRows,
      candidateLabels,
      entailmentId = 0,
      contradictionId = 1,
      batchSize = 8,
      maxSentenceLength = 512,
      caseSensitive = true,
      coalesceSentences = false,
      tags = Map.empty,
      ActivationFunction.softmax)

    assert(result.length == 2)
    assert(result.forall(_.isEmpty))
  }
}
