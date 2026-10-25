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

/** Padding-specific coverage for `predictSpanGrouped`.
  *
  * The sibling spec (`XXXForClassificationPredictSpanGroupedTestSpec`) checks ordering and
  * regrouping, but it cannot say anything about padding for two reasons: its fake emits exactly
  * one token per document, so every encoded row comes out the same width and no padding is ever
  * added; and its fake `tagSpan` derives its peak from `lastIndexWhere(_ != sentencePadTokenId)`,
  * i.e. it models a backend that already ignores padding perfectly. A real backend does not:
  * `tagSpan` softmaxes across the whole padded width, and several concrete `tagSpan`
  * implementations historically built an all-ones (or hardcoded-`0`-compared) attention mask, so
  * padding positions carry real probability mass and can win an argmax outright.
  *
  * This spec therefore does the opposite on both counts:
  *
  *   - documents tokenize to a length derived from their own text, so a batch genuinely mixes
  *     widths and shorter rows really are padded;
  *   - the fake `tagSpan` is deliberately padding-NAIVE - it scores strictly increasing with
  *     position, so the highest-scoring slot in a padded row is always the LAST PAD. Any
  *     implementation that does not confine the argmax to the row's own unpadded region will pick
  *     a padding position and fail these tests.
  */
class XXXForClassificationPredictSpanPaddingTestSpec extends AnyFlatSpec {

  class PaddingNaiveSpanClassification extends XXXForClassification {
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

    /** Emits one token per character of the document's text, so documents of different lengths
      * produce encoded rows of different widths - which is what forces real padding once several
      * rows share a batch.
      */
    override def tokenizeDocument(
        docs: Seq[Annotation],
        maxSeqLength: Int,
        caseSensitive: Boolean): Seq[WordpieceTokenizedSentence] =
      docs.map { doc =>
        val pieces = doc.result.zipWithIndex.map { case (ch, i) =>
          TokenPiece(
            wordpiece = ch.toString,
            token = ch.toString,
            pieceId = doc.begin + i + 1,
            isWordStart = true,
            begin = doc.begin + i,
            end = doc.begin + i)
        }.toArray
        WordpieceTokenizedSentence(pieces)
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

    /** Deliberately padding-naive: the raw logit increases with position, so in a padded row the
      * highest-scoring slot is always the final PAD. Mirrors a backend whose attention mask does
      * not exclude padding.
      *
      * Softmaxed across the full padded width, exactly as every real `tagSpan` does before
      * returning - that is what makes "restrict to the valid prefix and renormalise" equivalent
      * to having never padded at all, and it is the property the invariance test below pins down.
      */
    override def tagSpan(batch: Seq[Array[Int]]): (Array[Array[Float]], Array[Array[Float]]) = {
      val maxLen = batch.map(_.length).max

      def softmax(logits: Array[Float]): Array[Float] = {
        val shifted = logits.map(_ - logits.max) // max-subtraction for numerical stability
        val exps = shifted.map(x => math.exp(x.toDouble))
        val total = exps.sum
        exps.map(e => (e / total).toFloat)
      }

      val scores =
        batch.map(_ => softmax(Array.tabulate(maxLen)(i => (i + 1).toFloat))).toArray
      (scores, scores.map(_.clone()))
    }

    override def findIndexedToken(
        tokenizedSentences: Seq[TokenizedSentence],
        sentence: (WordpieceTokenizedSentence, Int),
        tokenPiece: TokenPiece): Option[IndexedToken] = None
  }

  /** Rows of [question, context] whose text lengths vary widely, guaranteeing that any batch
    * larger than one mixes widths and therefore pads.
    */
  private def mixedWidthRows(): Seq[Seq[Annotation]] = {
    val widths = Seq(1, 7, 2, 12, 3, 20, 1, 9, 4, 15)
    var cursor = 0
    widths.map { w =>
      def nextDoc(len: Int): Annotation = {
        val begin = cursor
        val end = begin + len - 1
        cursor += len + 2
        Annotation(AnnotatorType.DOCUMENT, begin, end, "x" * len, Map.empty)
      }
      Seq(nextDoc(2), nextDoc(w)) // short question, variable-width context
    }
  }

  /** Each row's own unpadded encoded width, per the trait's own encodeSequence. */
  private def encodedWidth(model: PaddingNaiveSpanClassification, row: Seq[Annotation]): Int = {
    val q = model.tokenizeDocument(Seq(row.head), 512, caseSensitive = true)
    val c = model.tokenizeDocument(row.drop(1), 512, caseSensitive = true)
    model.encodeSequence(q, c, 512).head.length
  }

  Seq(2, 3, 5, 10).foreach { batchSize =>
    it should s"never select a padding position as the answer span with batchSize=$batchSize" taggedAs FastTest in {
      val model = new PaddingNaiveSpanClassification
      val rows = mixedWidthRows()

      val actual =
        model.predictSpanGrouped(
          rows,
          batchSize = batchSize,
          maxSentenceLength = 512,
          caseSensitive = true)

      rows.zip(actual).foreach { case (row, annotations) =>
        val width = encodedWidth(model, row)
        annotations.foreach { a =>
          val start = a.metadata("start").toInt
          val end = a.metadata("end").toInt
          assert(
            start < width,
            s"start index $start landed in padding (row's own width is $width)")
          assert(end < width, s"end index $end landed in padding (row's own width is $width)")
        }
      }
    }
  }

  it should "produce identical answers and scores no matter which rows share a batch" taggedAs FastTest in {
    val model = new PaddingNaiveSpanClassification
    val rows = mixedWidthRows()

    // batchSize=1 is the unpadded ground truth: a batch of one is never padded.
    val unbatched =
      model.predictSpanGrouped(rows, batchSize = 1, maxSentenceLength = 512, caseSensitive = true)

    Seq(2, 3, 5, 10).foreach { batchSize =>
      val batched =
        model.predictSpanGrouped(
          rows,
          batchSize = batchSize,
          maxSentenceLength = 512,
          caseSensitive = true)

      assert(
        batched.map(_.map(_.result)) == unbatched.map(_.map(_.result)),
        s"answer text changed with batchSize=$batchSize")

      unbatched.zip(batched).foreach { case (expectedRow, actualRow) =>
        expectedRow.zip(actualRow).foreach { case (expected, actual) =>
          // Positions are exact integers.
          Seq("start", "end").foreach { key =>
            assert(
              expected.metadata(key) == actual.metadata(key),
              s"metadata '$key' changed with batchSize=$batchSize: " +
                s"${expected.metadata(key)} -> ${actual.metadata(key)}")
          }
          // Scores are mathematically identical but go through a renormalisation, so compare
          // numerically rather than by their rendered string.
          Seq("score", "start_score", "end_score").foreach { key =>
            val e = expected.metadata(key).toFloat
            val a = actual.metadata(key).toFloat
            assert(
              math.abs(e - a) < 1e-5f,
              s"metadata '$key' changed with batchSize=$batchSize: $e -> $a")
          }
        }
      }
    }
  }
}
