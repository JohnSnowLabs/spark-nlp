/*
 * Copyright 2017-2023 John Snow Labs
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

import com.johnsnowlabs.ml.onnx.OnnxWrapper
import com.johnsnowlabs.ml.openvino.OpenvinoWrapper
import com.johnsnowlabs.ml.tensorflow.TensorflowWrapper
import com.johnsnowlabs.nlp.annotators.common.WordpieceTokenizedSentence
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}

private[johnsnowlabs] class ZeroShotNerClassification(
    override val tensorflowWrapper: Option[TensorflowWrapper],
    override val onnxWrapper: Option[OnnxWrapper],
    override val openvinoWrapper: Option[OpenvinoWrapper],
    override val sentenceStartTokenId: Int,
    override val sentenceEndTokenId: Int,
    override val sentencePadTokenId: Int,
    val handleImpossibleAnswer: Boolean = false,
    configProtoBytes: Option[Array[Byte]] = None,
    tags: Map[String, Int],
    signatures: Option[Map[String, String]] = None,
    merges: Map[(String, String), Int],
    vocabulary: Map[String, Int])
    extends RoBertaClassification(
      tensorflowWrapper,
      onnxWrapper,
      openvinoWrapper,
      sentenceStartTokenId,
      sentenceEndTokenId,
      sentencePadTokenId,
      configProtoBytes,
      tags,
      signatures,
      merges,
      vocabulary) {

  override def tagSpan(batch: Seq[Array[Int]]): (Array[Array[Float]], Array[Array[Float]]) = {
    val (startLogits, endLogits) = super.tagSpan(batch)
    val contextStartOffsets = batch.map(_.indexOf(sentenceEndTokenId))

    /** Zeroes out the question, and the `</s>` that closes it, so they cannot win the argmax.
      *
      * The second clause guards a row that has no context at all (its lone `</s>` is the last
      * token in the row). It must be compared against the ROW's own width - comparing it against
      * `startLogits.length` (the number of rows in the batch, i.e. `batchSize`) as this used to
      * do was harmless only because [[predictSpan]] used to call this with a batch of exactly
      * one, making that comparison equivalent to `contextStart == 0`. With real batches (see
      * [[predictSpanGrouped]]) that comparison instead fires whenever a row's `</s>` happens to
      * land at index `batchSize - 1`, unrelated to that row's own length, spuriously zeroing a
      * perfectly normal row's scores depending on which other rows share its batch.
      */
    def maskQuestion(scores: Array[Array[Float]]): Array[Array[Float]] =
      scores.zip(contextStartOffsets).map { case (row, contextStart) =>
        row.zipWithIndex.map { case (score, i) =>
          if (((i > 0) && i <= contextStart) || (contextStart == row.length - 1)) 0f else score
        }
      }

    (maskQuestion(startLogits), maskQuestion(endLogits))
  }

  override def predictSpan(
      documents: Seq[Annotation],
      maxSentenceLength: Int,
      caseSensitive: Boolean,
      mergeTokenStrategy: String,
      engine: String): Seq[Annotation] =
    predictSpanGrouped(
      Seq(documents),
      batchSize = 1,
      maxSentenceLength,
      caseSensitive,
      mergeTokenStrategy,
      engine).head

  /** One row's worth of input to [[predictSpanGrouped]]. Mirrors the trait's own `SpanExample`,
    * plus the row's context annotation - this class needs it to turn a token span back into
    * character offsets in the original document.
    */
  private case class ZeroShotSpanExample(
      rowIndex: Int,
      contextAnnot: Seq[Annotation],
      wordPieceTokenizedQuestion: Seq[WordpieceTokenizedSentence],
      wordPieceTokenizedContext: Seq[WordpieceTokenizedSentence],
      encoded: Array[Int])

  /** Batched counterpart of [[predictSpan]], and now the single implementation of both.
    *
    * `RoBertaForQuestionAnswering.batchAnnotate` (via `ZeroShotNerModel`) calls
    * `predictSpanGrouped` rather than `predictSpan` since SPARKNLP-45. Without this override
    * ZeroShotNer silently fell through to `RoBertaClassification.predictSpanGrouped`, which
    * decodes with plain RoBERTa QA semantics - a different score (average, not what
    * `RoBertaClassification` itself does) and, critically, `begin = 0 / end = content.length - 1`
    * instead of the answer's actual character offsets in the context.
    * `ZeroShotNerModel.isTokenInEntity` matches NER tokens against exactly those character
    * offsets, so the missing override turned every prediction into either the wrong entity span
    * or no entity at all.
    *
    * Two things [[predictSpan]] got away with only because it always ran one example at a time:
    *
    *   1. `tagSpan`'s ONNX path (in the inherited `RoBertaClassification`) builds a rectangular
    *      tensor out of the raw encoded rows, so every row in a batch has to be the same length -
    *      trivially true for a batch of one. Rows are padded here to their batch's own max length
    *      with `sentencePadTokenId`.
    *   1. `tagSpan` softmaxes across the full padded width, so a short row's scores would be
    *      diluted by whichever longer row shared its batch, and its argmax could even land on a
    *      padding position. Scores are confined to the row's own unpadded width before being
    *      renormalised, which is exactly equivalent to softmaxing over the unpadded row - so a
    *      row's answer does not depend on what it was batched with.
    */
  override def predictSpanGrouped(
      rowsOfDocuments: Seq[Seq[Annotation]],
      batchSize: Int,
      maxSentenceLength: Int,
      caseSensitive: Boolean,
      mergeTokenStrategy: String,
      engine: String): Seq[Seq[Annotation]] = {

    val examples: Seq[ZeroShotSpanExample] = rowsOfDocuments.zipWithIndex.flatMap {
      case (documents, rowIndex) =>
        if (documents.isEmpty) None
        else {
          val questionAnnot = Seq(documents.head)
          val contextAnnot = documents.drop(1)

          val wordPieceTokenizedQuestion =
            tokenizeDocument(questionAnnot, maxSentenceLength, caseSensitive)
          val wordPieceTokenizedContext =
            tokenizeDocument(contextAnnot, maxSentenceLength, caseSensitive)
          val encoded = encodeSequence(
            wordPieceTokenizedQuestion,
            wordPieceTokenizedContext,
            maxSentenceLength).head

          Some(
            ZeroShotSpanExample(
              rowIndex,
              contextAnnot,
              wordPieceTokenizedQuestion,
              wordPieceTokenizedContext,
              encoded))
        }
    }

    if (examples.isEmpty) return rowsOfDocuments.map(_ => Seq.empty[Annotation])

    val resultsWithRow: Seq[(Annotation, Int)] =
      batchByLength[ZeroShotSpanExample, (Annotation, Int)](
        examples,
        batchSize,
        _.encoded.length) { batch =>
        val maxLen = batch.map(_.encoded.length).max
        val paddedEncoded = batch.map { example =>
          example.encoded ++ Array.fill(maxLen - example.encoded.length)(sentencePadTokenId)
        }
        val (startLogits, endLogits) = tagSpan(paddedEncoded)

        batch.zipWithIndex.map { case (example, i) =>
          (
            decodeSpan(example, startLogits(i), endLogits(i), mergeTokenStrategy),
            example.rowIndex)
        }
      }

    val byRow =
      resultsWithRow.groupBy(_._2).map { case (rowIndex, pairs) => rowIndex -> pairs.map(_._1) }
    rowsOfDocuments.indices.map(rowIndex => byRow.getOrElse(rowIndex, Seq.empty[Annotation]))
  }

  /** Renormalises so the retained scores sum to 1 again, as `predictSpan` always did - `tagSpan`
    * has already zeroed the question positions out of an otherwise-normalised softmax row.
    */
  private def renormalize(scores: Array[Float]): Array[Float] = {
    val total = scores.sum
    if (total <= 0f) scores else scores.map(_ / total)
  }

  private def decodeSpan(
      example: ZeroShotSpanExample,
      rawStartScores: Array[Float],
      rawEndScores: Array[Float],
      mergeTokenStrategy: String): Annotation = {

    val contextAnnot = example.contextAnnot
    // This row's own unpadded width - the batch may be wider.
    val validLength = example.encoded.length
    val startScores = renormalize(rawStartScores.take(validLength))
    val endScores = renormalize(rawEndScores.take(validLength))

    val startIndex =
      startScores.zipWithIndex.drop(if (handleImpossibleAnswer) 0 else 1).maxBy(_._1)
    val endIndex = endScores.zipWithIndex.drop(if (handleImpossibleAnswer) 0 else 1).maxBy(_._1)

    val allTokenPieces =
      example.wordPieceTokenizedQuestion.head.tokens ++
        example.wordPieceTokenizedContext.flatMap(x => x.tokens)
    val decodedAnswer = allTokenPieces.slice(startIndex._2 - 3, endIndex._2 - 2)
    // Check if the answer span starts at the CLS symbol 0 - if so return empty string
    val content =
      if (startIndex._2 > 0)
        mergeTokenStrategy match {
          case MergeTokenStrategy.vocab =>
            decodedAnswer.filter(_.isWordStart).map(x => x.token).mkString(" ")
          case MergeTokenStrategy.sentencePiece =>
            val token = ""
            decodedAnswer
              .map(x =>
                if (x.isWordStart) " " + token + x.token
                else token + x.token)
              .mkString("")
              .trim
        }
      else ""

    if (content.isEmpty) {
      Annotation(
        annotatorType = AnnotatorType.CHUNK,
        begin = 0,
        end = 0,
        result = content,
        metadata = Map(
          "sentence" -> contextAnnot.head.metadata.getOrElse("sentence", "0"),
          "chunk" -> "0",
          "start" -> "0",
          "start_score" -> "0",
          "end" -> "0",
          "end_score" -> "0",
          "score" -> "0",
          "start_char" -> "0",
          "end_char" -> "0"))
    } else {
      val sentenceOffset = contextAnnot.head.begin
      val tokenStartAdjustment =
        if (contextAnnot.head.result(decodedAnswer.head.begin - sentenceOffset) == ' ') 1 else 0
      val startPos = decodedAnswer.head.begin + tokenStartAdjustment
      val endPos = decodedAnswer.last.end
      Annotation(
        annotatorType = AnnotatorType.CHUNK,
        begin = startPos,
        end = endPos,
        result = content,
        metadata = Map(
          "sentence" -> contextAnnot.head.metadata.getOrElse("sentence", "0"),
          "score" -> ((startIndex._1 + endIndex._1) / 2).toString))
    }
  }

}
