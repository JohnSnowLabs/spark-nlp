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

import com.johnsnowlabs.ml.tensorflow.TensorflowWrapper
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}

private[johnsnowlabs] class ZeroShotNerClassification(
    override val tensorflowWrapper: TensorflowWrapper,
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

    (
      startLogits
        .zip(contextStartOffsets)
        .map(x =>
          x._1.zipWithIndex.map(y =>
            if (((y._2 > 0) && y._2 <= x._2) || (x._2 == startLogits.length - 1)) 0f else y._1)),
      endLogits
        .zip(contextStartOffsets)
        .map(x =>
          x._1.zipWithIndex.map(y =>
            if (((y._2 > 0) && y._2 <= x._2) || (x._2 == startLogits.length - 1)) 0f else y._1)))
  }

  override def predictSpan(
      documents: Seq[Annotation],
      maxSentenceLength: Int,
      caseSensitive: Boolean,
      mergeTokenStrategy: String,
      engine: String): Seq[Annotation] = {
    val questionAnnot = Seq(documents.head)
    val contextAnnot = documents.drop(1)

    val wordPieceTokenizedQuestion =
      tokenizeDocument(questionAnnot, maxSentenceLength, caseSensitive)
    val wordPieceTokenizedContext =
      tokenizeDocument(contextAnnot, maxSentenceLength, caseSensitive)

    val encodedInput =
      encodeSequence(wordPieceTokenizedQuestion, wordPieceTokenizedContext, maxSentenceLength)
    val (startLogits, endLogits) = tagSpan(encodedInput)

    val startScores = startLogits.map(x => x.map(y => y / x.sum)).head
    val endScores = endLogits.map(x => x.map(y => y / x.sum)).head

    val startIndex =
      startScores.zipWithIndex.drop(if (handleImpossibleAnswer) 0 else 1).maxBy(_._1)
    val endIndex = endScores.zipWithIndex.drop(if (handleImpossibleAnswer) 0 else 1).maxBy(_._1)

    val allTokenPieces =
      wordPieceTokenizedQuestion.head.tokens ++ wordPieceTokenizedContext.flatMap(x => x.tokens)
    val decodedAnswer = allTokenPieces.slice(startIndex._2 - 2, endIndex._2 - 1)
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
      Seq(
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
            "end_char" -> "0")))
    } else {
      val sentenceOffset = contextAnnot.head.begin
      val tokenStartAdjustment =
        if (contextAnnot.head.result(decodedAnswer.head.begin - sentenceOffset) == ' ') 1 else 0
      val startPos = decodedAnswer.head.begin + tokenStartAdjustment
      val endPos = decodedAnswer.last.end
      Seq(
        Annotation(
          annotatorType = AnnotatorType.CHUNK,
          begin = startPos,
          end = endPos,
          result = content,
          metadata = Map(
            "sentence" -> contextAnnot.head.metadata.getOrElse("sentence", "0"),
            "score" -> ((startIndex._1 + endIndex._1) / 2).toString)))
    }

  }
}
