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

import com.johnsnowlabs.ml.util.TensorFlow
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.{ActivationFunction, Annotation, AnnotatorType}

private[johnsnowlabs] object XXXForClassification {

  /** Mirrors HuggingFace's `PreTrainedTokenizer.clean_up_tokenization`: undoes the extra spaces a
    * naive word-start-token join introduces around punctuation and contractions (`"Levi ' s
    * Stadium"` -> `"Levi's Stadium"`).
    */
  def cleanUpTokenizationSpaces(text: String): String =
    text
      .replace(" .", ".")
      .replace(" ?", "?")
      .replace(" !", "!")
      .replace(" ,", ",")
      .replace(" ' ", "'")
      .replace(" n't", "n't")
      .replace(" 'm", "'m")
      .replace(" 's", "'s")
      .replace(" 've", "'ve")
      .replace(" 're", "'re")

  def joinWordPieces(pieces: Seq[TokenPiece], mergeTokenStrategy: String): String = {
    val joined = mergeTokenStrategy match {
      case MergeTokenStrategy.vocab =>
        pieces.filter(_.isWordStart).map(_.token).mkString(" ")
      case MergeTokenStrategy.sentencePiece =>
        pieces
          .map(x => if (x.isWordStart) " " + x.token else x.token)
          .mkString("")
          .trim
    }
    cleanUpTokenizationSpaces(joined)
  }
}

private[johnsnowlabs] trait XXXForClassification {

  protected val sentencePadTokenId: Int
  protected val sentenceStartTokenId: Int
  protected val sentenceEndTokenId: Int
  protected val sigmoidThreshold: Float

  def predict(
      tokenizedSentences: Seq[TokenizedSentence],
      batchSize: Int,
      maxSentenceLength: Int,
      caseSensitive: Boolean,
      tags: Map[String, Int]): Seq[Annotation] = {

    val wordPieceTokenizedSentences =
      tokenizeWithAlignment(tokenizedSentences, maxSentenceLength, caseSensitive)

    /*Run calculation by batches*/
    wordPieceTokenizedSentences.zipWithIndex
      .grouped(batchSize)
      .flatMap { batch =>
        val encoded = encode(batch, maxSentenceLength)
        val logits = tag(encoded)

        /*Combine tokens and calculated logits*/
        batch.zip(logits).flatMap { case (sentence, tokenVectors) =>
          val tokenLength = sentence._1.tokens.length

          /*All wordpiece logits*/
          val tokenLogits: Array[Array[Float]] = tokenVectors.slice(1, tokenLength + 1)

          val labelsWithScores = wordAndSpanLevelAlignmentWithTokenizer(
            tokenLogits,
            tokenizedSentences,
            sentence,
            tags)
          labelsWithScores
        }
      }
      .toSeq

  }

  /** Restricts a span score row produced over a padded batch back to the example's own unpadded
    * length, renormalising so the retained scores sum to 1 again.
    *
    * `tagSpan` softmaxes across the full padded width, so padding positions hold a share of the
    * probability mass and are themselves eligible for an argmax. Dropping them and dividing by
    * the retained sum is exactly equivalent to having softmaxed over the unpadded row alone
    * (`exp(x_i) / sum_{j in valid} exp(x_j)`), which keeps a row's answer and reported scores
    * independent of whichever other rows happened to share its batch.
    */
  protected def unpaddedScores(scores: Array[Float], validLength: Int): Array[Float] = {
    if (validLength >= scores.length) return scores
    val valid = scores.take(validLength)
    val total = valid.sum
    if (total <= 0f) valid else valid.map(_ / total)
  }

  /** Groups items into batches of similar length (to minimise padding waste, since `encode` pads
    * every item in a batch to that batch's own max length) then restores the caller's original
    * ordering once inference is done.
    *
    * `f` is handed one length-homogeneous batch and MUST return exactly one result per input
    * element, in the same order as that batch.
    */
  protected def batchByLength[T, R](items: Seq[T], batchSize: Int, lengthOf: T => Int)(
      f: Seq[T] => Seq[R]): Seq[R] = {
    items.zipWithIndex
      .sortBy { case (item, _) => lengthOf(item) }
      .grouped(batchSize)
      .flatMap { batch => f(batch.map(_._1)).zip(batch.map(_._2)) }
      .toSeq
      .sortBy(_._2)
      .map(_._1)
  }

  /** Batches token-classification inference across every sentence from every row in ONE pass,
    * instead of the caller invoking [[predict]] once per row - which would otherwise leave
    * `batchSize` capped at a single row's sentence count (typically 1-3), making it a no-op for
    * the common case.
    *
    * `tokenizedSentences` is still supplied per row (not flattened) because
    * [[wordAndSpanLevelAlignmentWithTokenizer]] / [[findIndexedToken]] resolve a sentence's
    * original tokens by indexing directly into that row's own list
    * (`tokenizedSentences(sentence._2)`) - that indexing must stay row-local, so only the
    * inference step is batched across rows, not the alignment step.
    *
    * @return
    *   one `Seq[Annotation]` per row, in the same order as `rowsOfTokenizedSentences`, with each
    *   row's own sentence/token order preserved regardless of how batches were grouped for
    *   inference.
    */
  def predictGrouped(
      rowsOfTokenizedSentences: Seq[Seq[TokenizedSentence]],
      batchSize: Int,
      maxSentenceLength: Int,
      caseSensitive: Boolean,
      tags: Map[String, Int]): Seq[Seq[Annotation]] = {

    // (wordpiece-tokenized sentence, its row index, its row-LOCAL sentence index)
    val items: Seq[(WordpieceTokenizedSentence, Int, Int)] =
      rowsOfTokenizedSentences.zipWithIndex.flatMap { case (tokenizedSentences, rowIndex) =>
        tokenizeWithAlignment(tokenizedSentences, maxSentenceLength, caseSensitive).zipWithIndex
          .map { case (wordpieceTokenizedSentence, localIndex) =>
            (wordpieceTokenizedSentence, rowIndex, localIndex)
          }
      }

    if (items.isEmpty) return rowsOfTokenizedSentences.map(_ => Seq.empty[Annotation])

    // one Seq[Annotation] (that single sentence's own token annotations) per item, in items' order
    val perSentenceAnnotations: Seq[Seq[Annotation]] =
      batchByLength[(WordpieceTokenizedSentence, Int, Int), Seq[Annotation]](
        items,
        batchSize,
        { case (wordpieceTokenizedSentence, _, _) => wordpieceTokenizedSentence.tokens.length }) {
        batch =>
          // encode's second tuple element is unused by encode itself; it only exists so callers
          // of `predict` can carry a position through, which we don't need here.
          val encoded = encode(batch.map { case (wpts, _, _) => (wpts, 0) }, maxSentenceLength)
          val logits = tag(encoded)
          batch.zip(logits).map {
            case ((wordpieceTokenizedSentence, rowIndex, localIndex), tokenVectors) =>
              val tokenLength = wordpieceTokenizedSentence.tokens.length
              val tokenLogits: Array[Array[Float]] = tokenVectors.slice(1, tokenLength + 1)
              wordAndSpanLevelAlignmentWithTokenizer(
                tokenLogits,
                rowsOfTokenizedSentences(rowIndex),
                (wordpieceTokenizedSentence, localIndex),
                tags)
          }
      }

    // Accumulate into per-row builders rather than repeatedly `++`-ing an immutable Seq, which
    // would be quadratic in a row's sentence count.
    val byRow =
      Array.fill(rowsOfTokenizedSentences.length)(Seq.newBuilder[Annotation])
    items.zip(perSentenceAnnotations).foreach { case ((_, rowIndex, _), sentenceAnnotations) =>
      byRow(rowIndex) ++= sentenceAnnotations
    }
    byRow.map(_.result()).toSeq
  }

  def predictSequence(
      tokenizedSentences: Seq[TokenizedSentence],
      sentences: Seq[Sentence],
      batchSize: Int,
      maxSentenceLength: Int,
      caseSensitive: Boolean,
      coalesceSentences: Boolean = false,
      tags: Map[String, Int],
      activation: String = ActivationFunction.softmax): Seq[Annotation] = {

    val wordPieceTokenizedSentences =
      tokenizeWithAlignment(tokenizedSentences, maxSentenceLength, caseSensitive)

    if (sentences.isEmpty) return Seq.empty[Annotation]

    /* Stage 1: run inference batch by batch, pairing each sentence with its OWN batch's
     * logits (not the full outer `sentences` list, which would misalign past the first batch) */
    val sentencesWithScores: Seq[(Sentence, Array[Float])] =
      wordPieceTokenizedSentences
        .zip(sentences)
        .zipWithIndex
        .grouped(batchSize)
        .flatMap { batch =>
          val tokensBatch = batch.map(x => (x._1._1, x._2))
          val encoded = encode(tokensBatch, maxSentenceLength)
          val logits = tagSequence(encoded, activation)
          batch.map(_._1._2).zip(logits)
        }
        .toSeq

    /* Stage 2: aggregate once over every sentence collected across all batches */
    aggregateSequenceScores(
      sentencesWithScores,
      coalesceSentences,
      activation,
      tags,
      sentences.head,
      sigmoidThreshold)

  }

  /** Batches sequence-classification inference across every sentence from every row in ONE pass,
    * instead of the caller invoking [[predictSequence]] once per row - which would otherwise
    * leave `batchSize` capped at a single row's sentence count.
    *
    * Unlike token classification, sequence classification doesn't need row-local positional
    * indexing - each result is anchored to its own `Sentence` object (which already carries its
    * row-local `.index`/`.start`/`.end`), so sentences can be freely flattened across rows for
    * inference. The one thing that DOES need to stay row-scoped is `coalesceSentences`: it means
    * "average every sentence into one annotation for the document", and once inference spans
    * multiple rows in a single call, "the document" is no longer implicitly "the call" - it has
    * to mean "the row". So scores are grouped back by row before calling the (unmodified)
    * [[aggregateSequenceScores]] once per row, rather than once for the whole batch.
    *
    * @return
    *   one `Seq[Annotation]` per row, in the same order as `rowsOfTokenizedSentences`.
    */
  def predictSequenceGrouped(
      rowsOfTokenizedSentences: Seq[Seq[TokenizedSentence]],
      rowsOfSentences: Seq[Seq[Sentence]],
      batchSize: Int,
      maxSentenceLength: Int,
      caseSensitive: Boolean,
      coalesceSentences: Boolean = false,
      tags: Map[String, Int],
      activation: String = ActivationFunction.softmax): Seq[Seq[Annotation]] = {

    // (wordpiece-tokenized sentence, its own Sentence, its row index)
    val items: Seq[(WordpieceTokenizedSentence, Sentence, Int)] =
      rowsOfTokenizedSentences.zip(rowsOfSentences).zipWithIndex.flatMap {
        case ((tokenizedSentences, sentences), rowIndex) =>
          tokenizeWithAlignment(tokenizedSentences, maxSentenceLength, caseSensitive)
            .zip(sentences)
            .map { case (wordpieceTokenizedSentence, sentence) =>
              (wordpieceTokenizedSentence, sentence, rowIndex)
            }
      }

    if (items.isEmpty) return rowsOfTokenizedSentences.map(_ => Seq.empty[Annotation])

    // Stage 1: run inference batch by batch (length-bucketed), carrying each sentence's own row
    // index through so coalescing can be resolved per row afterward.
    val sentencesWithScoresAndRow: Seq[(Sentence, Array[Float], Int)] =
      batchByLength[(WordpieceTokenizedSentence, Sentence, Int), (Sentence, Array[Float], Int)](
        items,
        batchSize,
        { case (wordpieceTokenizedSentence, _, _) => wordpieceTokenizedSentence.tokens.length }) {
        batch =>
          val encoded = encode(batch.map { case (wpts, _, _) => (wpts, 0) }, maxSentenceLength)
          val logits = tagSequence(encoded, activation)
          batch.zip(logits).map { case ((_, sentence, rowIndex), scores) =>
            (sentence, scores, rowIndex)
          }
      }

    // Stage 2: aggregate per row, so coalesceSentences averages within a row, not across rows
    val byRow = sentencesWithScoresAndRow.groupBy(_._3)
    rowsOfTokenizedSentences.indices.map { rowIndex =>
      byRow.get(rowIndex) match {
        case None => Seq.empty[Annotation]
        case Some(rowItems) =>
          val sentencesWithScores = rowItems.map { case (sentence, scores, _) =>
            (sentence, scores)
          }
          aggregateSequenceScores(
            sentencesWithScores,
            coalesceSentences,
            activation,
            tags,
            sentencesWithScores.head._1,
            sigmoidThreshold)
      }
    }
  }

  def predictSequenceWithZeroShot(
      tokenizedSentences: Seq[TokenizedSentence],
      sentences: Seq[Sentence],
      candidateLabels: Array[String],
      entailmentId: Int,
      contradictionId: Int,
      batchSize: Int,
      maxSentenceLength: Int,
      caseSensitive: Boolean,
      coalesceSentences: Boolean = false,
      tags: Map[String, Int],
      activation: String = ActivationFunction.softmax): Seq[Annotation] = {

    val wordPieceTokenizedSentences =
      tokenizeWithAlignment(tokenizedSentences, maxSentenceLength, caseSensitive)

    if (sentences.isEmpty) return Seq.empty[Annotation]

    val candidateLabelsKeyValue = candidateLabels.zipWithIndex.toMap
    val contradiction_id: Int = if (entailmentId == 0) contradictionId else 0

    val labelsTokenized =
      tokenizeSeqString(candidateLabels, maxSentenceLength, caseSensitive)

    /* Stage 1: run inference batch by batch, pairing each sentence with its OWN batch's
     * scores (not the full outer `sentences` list, which would misalign past the first batch) */
    val sentencesWithScores: Seq[(Sentence, Array[Float])] =
      wordPieceTokenizedSentences
        .zip(sentences)
        .zipWithIndex
        .grouped(batchSize)
        .flatMap { batch =>
          val tokensBatch = batch.map(x => (x._1._1, x._2))

          /* Start internal batching for zero shot */
          val encodedTokensLabels = tokensBatch.map { sent =>
            labelsTokenized.flatMap(labels =>
              encodeSequence(Seq(sent._1), Seq(labels), maxSentenceLength))
          }

          val logits = encodedTokensLabels.map { encodedSeq =>
            tagZeroShotSequence(encodedSeq, entailmentId, contradictionId, activation)
          }

          val multiClassScores =
            logits.map(scores => calculateSoftmax(scores.map(x => x(entailmentId))))
          val multiLabelScores =
            logits
              .map(scores =>
                scores
                  .map(x => calculateSoftmax(Array(x(contradiction_id), x(entailmentId))))
                  .map(_.last))

          val scoresForActivation = activation match {
            case ActivationFunction.softmax => multiClassScores
            case ActivationFunction.sigmoid => multiLabelScores
          }

          batch.map(_._1._2).zip(scoresForActivation)
        }
        .toSeq

    /* Stage 2: aggregate once over every sentence collected across all batches */
    aggregateSequenceScores(
      sentencesWithScores,
      coalesceSentences,
      activation,
      candidateLabelsKeyValue,
      sentences.head,
      sigmoidThreshold = 0.5f)

  }

  /** Batches zero-shot classification inference across every sentence from every row in ONE pass,
    * instead of the caller invoking [[predictSequenceWithZeroShot]] once per row.
    *
    * Same row-handling as [[predictSequenceGrouped]]: sentences flatten freely across rows for
    * inference (no positional-index constraint, since results are anchored to their own
    * `Sentence` object), and `coalesceSentences` is resolved per row via a groupBy before the
    * (unmodified) [[aggregateSequenceScores]], so "coalesce the document" means "coalesce the
    * row" even though a single inference batch may now span several rows.
    *
    * The per-sentence loop over candidate labels (`tagZeroShotSequence` called once per sentence,
    * batched only across that sentence's own labels) is unchanged from
    * [[predictSequenceWithZeroShot]] - flattening that into a single batch of sentence x label
    * pairs is a further optimisation left for a separate change, not bundled here.
    *
    * @return
    *   one `Seq[Annotation]` per row, in the same order as `rowsOfTokenizedSentences`.
    */
  def predictSequenceWithZeroShotGrouped(
      rowsOfTokenizedSentences: Seq[Seq[TokenizedSentence]],
      rowsOfSentences: Seq[Seq[Sentence]],
      candidateLabels: Array[String],
      entailmentId: Int,
      contradictionId: Int,
      batchSize: Int,
      maxSentenceLength: Int,
      caseSensitive: Boolean,
      coalesceSentences: Boolean = false,
      tags: Map[String, Int],
      activation: String = ActivationFunction.softmax): Seq[Seq[Annotation]] = {

    val candidateLabelsKeyValue = candidateLabels.zipWithIndex.toMap
    val contradiction_id: Int = if (entailmentId == 0) contradictionId else 0
    val labelsTokenized =
      tokenizeSeqString(candidateLabels, maxSentenceLength, caseSensitive)

    // (wordpiece-tokenized sentence, its own Sentence, its row index)
    val items: Seq[(WordpieceTokenizedSentence, Sentence, Int)] =
      rowsOfTokenizedSentences.zip(rowsOfSentences).zipWithIndex.flatMap {
        case ((tokenizedSentences, sentences), rowIndex) =>
          tokenizeWithAlignment(tokenizedSentences, maxSentenceLength, caseSensitive)
            .zip(sentences)
            .map { case (wordpieceTokenizedSentence, sentence) =>
              (wordpieceTokenizedSentence, sentence, rowIndex)
            }
      }

    if (items.isEmpty) return rowsOfTokenizedSentences.map(_ => Seq.empty[Annotation])

    val sentencesWithScoresAndRow: Seq[(Sentence, Array[Float], Int)] =
      batchByLength[(WordpieceTokenizedSentence, Sentence, Int), (Sentence, Array[Float], Int)](
        items,
        batchSize,
        { case (wordpieceTokenizedSentence, _, _) => wordpieceTokenizedSentence.tokens.length }) {
        batch =>
          /* Start internal batching for zero shot: one tagZeroShotSequence call per sentence,
           * batched across that sentence's own candidate labels */
          val encodedTokensLabels = batch.map { case (wordpieceTokenizedSentence, _, _) =>
            labelsTokenized.flatMap(labels =>
              encodeSequence(Seq(wordpieceTokenizedSentence), Seq(labels), maxSentenceLength))
          }

          val logits = encodedTokensLabels.map { encodedSeq =>
            tagZeroShotSequence(encodedSeq, entailmentId, contradictionId, activation)
          }

          val multiClassScores =
            logits.map(scores => calculateSoftmax(scores.map(x => x(entailmentId))))
          val multiLabelScores =
            logits.map(scores =>
              scores
                .map(x => calculateSoftmax(Array(x(contradiction_id), x(entailmentId))))
                .map(_.last))

          val scoresForActivation = activation match {
            case ActivationFunction.softmax => multiClassScores
            case ActivationFunction.sigmoid => multiLabelScores
          }

          batch
            .map { case (_, sentence, rowIndex) => (sentence, rowIndex) }
            .zip(scoresForActivation)
            .map { case ((sentence, rowIndex), scores) => (sentence, scores, rowIndex) }
      }

    val byRow = sentencesWithScoresAndRow.groupBy(_._3)
    rowsOfTokenizedSentences.indices.map { rowIndex =>
      byRow.get(rowIndex) match {
        case None => Seq.empty[Annotation]
        case Some(rowItems) =>
          val sentencesWithScores = rowItems.map { case (sentence, scores, _) =>
            (sentence, scores)
          }
          aggregateSequenceScores(
            sentencesWithScores,
            coalesceSentences,
            activation,
            candidateLabelsKeyValue,
            sentencesWithScores.head._1,
            sigmoidThreshold = 0.5f)
      }
    }
  }

  def scoresToLabelForSequenceClassifier(tags: Map[String, Int], scores: Array[Float]): String = {
    tags.find(_._2 == scores.zipWithIndex.maxBy(_._1)._2).map(_._1).getOrElse("NA")
  }

  def constructMetaForSequenceClassifier(
      tags: Map[String, Int],
      scores: Array[Float]): Array[(String, String)] = {
    scores.zipWithIndex.flatMap(x =>
      Map(tags.find(_._2 == x._2).map(_._1).getOrElse("NA") -> x._1.toString))
  }

  def constructAnnotationForSequenceClassifier(
      sentence: Sentence,
      label: String,
      meta: Array[(String, String)]): Annotation = {

    Annotation(
      annotatorType = AnnotatorType.CATEGORY,
      begin = sentence.start,
      end = sentence.end,
      result = label,
      metadata = Map("sentence" -> sentence.index.toString) ++ meta)

  }

  /** Aggregates per-sentence classification scores into annotations.
    *
    * When `coalesceSentences` is true, scores are averaged across every sentence passed in (i.e.
    * across the whole document, not per inference batch) and a single annotation is returned
    * anchored at `documentAnchor`. Otherwise one annotation is returned per sentence.
    *
    * Shared by [[predictSequence]] and [[predictSequenceWithZeroShot]] so both aggregate over the
    * full collected result set rather than per-batch.
    */
  protected def aggregateSequenceScores(
      sentencesWithScores: Seq[(Sentence, Array[Float])],
      coalesceSentences: Boolean,
      activation: String,
      tags: Map[String, Int],
      documentAnchor: Sentence,
      sigmoidThreshold: Float): Seq[Annotation] = {

    if (sentencesWithScores.isEmpty) return Seq.empty[Annotation]

    val allScores = sentencesWithScores.map(_._2).toArray

    activation match {
      case ActivationFunction.softmax =>
        if (coalesceSentences) {
          val scores = allScores.transpose.map(_.sum / allScores.length)
          val label = scoresToLabelForSequenceClassifier(tags, scores)
          val meta = constructMetaForSequenceClassifier(tags, scores)
          Seq(constructAnnotationForSequenceClassifier(documentAnchor, label, meta))
        } else {
          sentencesWithScores.map { case (sentence, scores) =>
            val label = scoresToLabelForSequenceClassifier(tags, scores)
            val meta = constructMetaForSequenceClassifier(tags, scores)
            constructAnnotationForSequenceClassifier(sentence, label, meta)
          }
        }

      case ActivationFunction.sigmoid =>
        if (coalesceSentences) {
          val scores = allScores.transpose.map(_.sum / allScores.length)
          val labels = scores.zipWithIndex
            .filter(x => x._1 > sigmoidThreshold)
            .flatMap(x => tags.filter(_._2 == x._2))
          val meta = constructMetaForSequenceClassifier(tags, scores)
          labels
            .map(label =>
              constructAnnotationForSequenceClassifier(documentAnchor, label._1, meta))
            .toSeq
        } else {
          sentencesWithScores.flatMap { case (sentence, scores) =>
            val labels = scores.zipWithIndex
              .filter(x => x._1 > sigmoidThreshold)
              .flatMap(x => tags.filter(_._2 == x._2))
            val meta = constructMetaForSequenceClassifier(tags, scores)
            labels.map(label =>
              constructAnnotationForSequenceClassifier(sentence, label._1, meta))
          }
        }
    }
  }

  def predictSpan(
      documents: Seq[Annotation],
      maxSentenceLength: Int,
      caseSensitive: Boolean,
      mergeTokenStrategy: String = MergeTokenStrategy.vocab,
      engine: String = TensorFlow.name): Seq[Annotation] = {

    val questionAnnot = Seq(documents.head)
    val contextAnnot = documents.drop(1)

    val wordPieceTokenizedQuestion =
      tokenizeDocument(questionAnnot, maxSentenceLength, caseSensitive)
    val wordPieceTokenizedContext =
      tokenizeDocument(contextAnnot, maxSentenceLength, caseSensitive)

    val encodedInput =
      encodeSequence(wordPieceTokenizedQuestion, wordPieceTokenizedContext, maxSentenceLength)
    val (startLogits, endLogits) = tagSpan(encodedInput)

    val startScores = startLogits.transpose.map(_.sum / startLogits.length)
    val endScores = endLogits.transpose.map(_.sum / endLogits.length)

    val startIndex = startScores.zipWithIndex.maxBy(_._1)
    val endIndex = endScores.zipWithIndex.maxBy(_._1)

    val offsetStartIndex = if (engine == TensorFlow.name) 2 else 1
    val offsetEndIndex = if (engine == TensorFlow.name) 1 else 0

    val allTokenPieces =
      wordPieceTokenizedQuestion.head.tokens ++ wordPieceTokenizedContext.flatMap(x => x.tokens)
    val decodedAnswer =
      allTokenPieces.slice(startIndex._2 - offsetStartIndex, endIndex._2 - offsetEndIndex)
    val content = XXXForClassification.joinWordPieces(decodedAnswer, mergeTokenStrategy)

    Seq(
      Annotation(
        annotatorType = AnnotatorType.CHUNK,
        begin = 0,
        end = if (content.isEmpty) 0 else content.length - 1,
        result = content,
        metadata = Map(
          "sentence" -> "0",
          "chunk" -> "0",
          "start" -> startIndex._2.toString,
          "start_score" -> startIndex._1.toString,
          "end" -> endIndex._2.toString,
          "end_score" -> endIndex._1.toString,
          "score" -> ((startIndex._1 + endIndex._1) / 2).toString)))

  }

  /** One row's worth of input to [[predictSpanGrouped]]: the row's own tokenized question/context
    * plus that row's (unpadded) encoded sequence, tagged with its row index.
    */
  protected case class SpanExample(
      rowIndex: Int,
      wordPieceTokenizedQuestion: Seq[WordpieceTokenizedSentence],
      wordPieceTokenizedContext: Seq[WordpieceTokenizedSentence],
      encoded: Array[Int])

  /** Batches question-answering span prediction across every row in ONE pass, instead of the
    * caller invoking [[predictSpan]] once per row (which has no `batchSize` at all today).
    *
    * Four things [[predictSpan]] gets away with only because it is always called with exactly one
    * example, which this method can no longer assume:
    *
    *   1. `startLogits.transpose.map(_.sum / startLogits.length)` looks like it "unwraps" a batch
    *      dimension, but it is actually an AVERAGE across the batch. With one example that
    *      average is a no-op; with several it would blend different questions' answer-position
    *      scores together. This method indexes `startScores(i)`/`endScores(i)` per example
    *      instead.
    *   1. `tagSpan`'s ONNX path builds a rectangular tensor from the raw encoded arrays
    *      (`OnnxTensor.createTensor` on a `batch.map(...).toArray)`), which requires every row to
    *      be the same length - fine for a batch of 1, which is trivially rectangular regardless
    *      of its own length, but not for a real multi-example batch. This method pads every
    *      example in a batch to that batch's own max length with `sentencePadTokenId` before
    *      calling `tagSpan`.
    *   1. Padding must use each model family's own `sentencePadTokenId` (NOT a hardcoded 0 - for
    *      example RoBERTa and MPNet use 1), because that is the value the concrete `tagSpan`
    *      implementations compare against when deriving their attention mask. Note this was NOT
    *      uniformly true before this method existed: several span paths hardcoded `x == 0` or
    *      built an all-ones mask, which was harmless only because a batch of 1 is never padded.
    *      Those were fixed alongside this method.
    *   1. `tagSpan` softmaxes over the FULL padded width, so a short example's scores would be
    *      diluted by whatever longer example shared its batch, and its argmax could even land on
    *      a padding position. This method restricts both to the example's own unpadded length and
    *      renormalises, which is exactly equivalent to softmaxing over the unpadded row - so a
    *      given row's answer and scores do not depend on which other rows it was batched with.
    *
    * @return
    *   one `Seq[Annotation]` per row, in the same order as `rowsOfDocuments`.
    */
  def predictSpanGrouped(
      rowsOfDocuments: Seq[Seq[Annotation]],
      batchSize: Int,
      maxSentenceLength: Int,
      caseSensitive: Boolean,
      mergeTokenStrategy: String = MergeTokenStrategy.vocab,
      engine: String = TensorFlow.name): Seq[Seq[Annotation]] = {

    val examples: Seq[SpanExample] = rowsOfDocuments.zipWithIndex.flatMap {
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
            SpanExample(rowIndex, wordPieceTokenizedQuestion, wordPieceTokenizedContext, encoded))
        }
    }

    if (examples.isEmpty) return rowsOfDocuments.map(_ => Seq.empty[Annotation])

    val resultsWithRow: Seq[(Annotation, Int)] =
      batchByLength[SpanExample, (Annotation, Int)](examples, batchSize, _.encoded.length) {
        batch =>
          val maxLen = batch.map(_.encoded.length).max
          val paddedEncoded = batch.map { example =>
            example.encoded ++ Array.fill(maxLen - example.encoded.length)(sentencePadTokenId)
          }
          val (startScores, endScores) = tagSpan(paddedEncoded)

          batch.zipWithIndex.map { case (example, i) =>
            // Confine argmax/scores to this example's own unpadded region - see point 4 above.
            val startIndex =
              unpaddedScores(startScores(i), example.encoded.length).zipWithIndex.maxBy(_._1)
            val endIndex =
              unpaddedScores(endScores(i), example.encoded.length).zipWithIndex.maxBy(_._1)

            val offsetStartIndex = if (engine == TensorFlow.name) 2 else 1
            val offsetEndIndex = if (engine == TensorFlow.name) 1 else 0

            val allTokenPieces =
              example.wordPieceTokenizedQuestion.head.tokens ++
                example.wordPieceTokenizedContext.flatMap(x => x.tokens)
            val decodedAnswer =
              allTokenPieces.slice(startIndex._2 - offsetStartIndex, endIndex._2 - offsetEndIndex)
            val content =
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

            val annotation = Annotation(
              annotatorType = AnnotatorType.CHUNK,
              begin = 0,
              end = if (content.isEmpty) 0 else content.length - 1,
              result = content,
              metadata = Map(
                "sentence" -> "0",
                "chunk" -> "0",
                "start" -> startIndex._2.toString,
                "start_score" -> startIndex._1.toString,
                "end" -> endIndex._2.toString,
                "end_score" -> endIndex._1.toString,
                "score" -> ((startIndex._1 + endIndex._1) / 2).toString))

            (annotation, example.rowIndex)
          }
      }

    val byRow =
      resultsWithRow.groupBy(_._2).map { case (rowIndex, pairs) => rowIndex -> pairs.map(_._1) }
    rowsOfDocuments.indices.map(rowIndex => byRow.getOrElse(rowIndex, Seq.empty[Annotation]))
  }

  def predictSpanMultipleChoice(
      documents: Seq[Annotation],
      choicesDelimiter: String,
      maxSentenceLength: Int,
      caseSensitive: Boolean): Seq[Annotation] = {

    val questionAnnotation = Seq(documents.head)
    val choices =
      documents.drop(1).flatMap(annotation => annotation.result.split(choicesDelimiter))

    val wordPieceTokenizedQuestions =
      tokenizeDocument(questionAnnotation, maxSentenceLength, caseSensitive)

    val inputIds = choices.flatMap { choice =>
      val choiceAnnotation =
        Seq(Annotation(AnnotatorType.DOCUMENT, 0, choice.length, choice, Map("sentence" -> "0")))
      val wordPieceTokenizedChoice =
        tokenizeDocument(choiceAnnotation, maxSentenceLength, caseSensitive)
      encodeSequenceWithPadding(
        wordPieceTokenizedQuestions,
        wordPieceTokenizedChoice,
        maxSentenceLength)
    }

    val scores = tagSpanMultipleChoice(inputIds)
    val (score, scoreIndex) = scores.zipWithIndex.maxBy(_._1)
    val prediction = choices(scoreIndex)

    Seq(
      Annotation(
        annotatorType = AnnotatorType.CHUNK,
        begin = 0,
        end = if (prediction.isEmpty) 0 else prediction.length - 1,
        result = prediction,
        metadata = Map("sentence" -> "0", "chunk" -> "0", "score" -> score.toString)))
  }

  def tokenizeWithAlignment(
      sentences: Seq[TokenizedSentence],
      maxSeqLength: Int,
      caseSensitive: Boolean): Seq[WordpieceTokenizedSentence]

  def tokenizeSeqString(
      candidateLabels: Seq[String],
      maxSeqLength: Int,
      caseSensitive: Boolean): Seq[WordpieceTokenizedSentence]

  def tokenizeDocument(
      docs: Seq[Annotation],
      maxSeqLength: Int,
      caseSensitive: Boolean): Seq[WordpieceTokenizedSentence]

  /** Encode the input sequence to indexes IDs adding padding where necessary */
  def encode(
      sentences: Seq[(WordpieceTokenizedSentence, Int)],
      maxSequenceLength: Int): Seq[Array[Int]] = {
    val maxSentenceLength =
      Array(
        maxSequenceLength - 2,
        sentences.map { case (wpTokSentence, _) =>
          wpTokSentence.tokens.length
        }.max).min

    sentences
      .map { case (wpTokSentence, _) =>
        val tokenPieceIds = wpTokSentence.tokens.map(t => t.pieceId)
        val padding = Array.fill(maxSentenceLength - tokenPieceIds.length)(sentencePadTokenId)

        Array(sentenceStartTokenId) ++ tokenPieceIds.take(maxSentenceLength) ++ Array(
          sentenceEndTokenId) ++ padding
      }
  }

  def encodeSequence(
      seq1: Seq[WordpieceTokenizedSentence],
      seq2: Seq[WordpieceTokenizedSentence],
      maxSequenceLength: Int): Seq[Array[Int]] = {

    val question = seq1
      .flatMap { wpTokSentence =>
        wpTokSentence.tokens.map(t => t.pieceId)
      }
      .toArray
      .take(maxSequenceLength - 2) ++ Array(sentenceEndTokenId)

    val context = seq2
      .flatMap { wpTokSentence =>
        wpTokSentence.tokens.map(t => t.pieceId)
      }
      .toArray
      .take(maxSequenceLength - question.length - 2) ++ Array(sentenceEndTokenId)

    Seq(Array(sentenceStartTokenId) ++ question ++ context)
  }

  def encodeSequenceWithPadding(
      seq1: Seq[WordpieceTokenizedSentence],
      seq2: Seq[WordpieceTokenizedSentence],
      maxSequenceLength: Int): Seq[Array[Int]] = {

    val question = seq1.flatMap { wpTokSentence =>
      wpTokSentence.tokens.map(t => t.pieceId)
    }.toArray

    val context = seq2.flatMap { wpTokSentence =>
      wpTokSentence.tokens.map(t => t.pieceId)
    }.toArray

    val availableLength = maxSequenceLength - 3 // (excluding special tokens)
    val truncatedQuestion = question.take(availableLength)
    val remainingLength = availableLength - truncatedQuestion.length
    val truncatedContext = context.take(remainingLength)

    val assembleSequence =
      Array(sentenceStartTokenId) ++ truncatedQuestion ++ Array(sentenceEndTokenId) ++
        truncatedContext ++ Array(sentenceEndTokenId)

    val paddingLength = maxSequenceLength - assembleSequence.length
    val paddedSequence = if (paddingLength > 0) {
      assembleSequence ++ Array.fill(paddingLength)(sentencePadTokenId)
    } else {
      assembleSequence
    }

    Seq(paddedSequence)
  }

  def tag(batch: Seq[Array[Int]]): Seq[Array[Array[Float]]]

  def tagSequence(batch: Seq[Array[Int]], activation: String): Array[Array[Float]]

  def tagZeroShotSequence(
      batch: Seq[Array[Int]],
      entailmentId: Int,
      contradictionId: Int,
      activation: String): Array[Array[Float]]

  def tagSpan(batch: Seq[Array[Int]]): (Array[Array[Float]], Array[Array[Float]])

  def tagSpanMultipleChoice(batch: Seq[Array[Int]]): Array[Float] = Array()

  /** Calculate softmax from returned logits
    * @param scores
    *   logits output from output layer
    * @return
    */
  def calculateSoftmax(scores: Array[Float]): Array[Float] = {
    val exp = scores.map(x => math.exp(x))
    exp.map(x => x / exp.sum).map(_.toFloat)
  }

  /** Calculate sigmoid from returned logits
    * @param scores
    *   logits output from output layer
    * @return
    */
  def calculateSigmoid(scores: Array[Float]): Array[Float] = {
    scores.map(x => 1 / (1 + Math.exp(-x)).toFloat)
  }

  /** Word-level and span-level alignment with Tokenizer
    * https://github.com/google-research/bert#tokenization
    *
    * ### Input orig_tokens = ["John", "Johanson", "'s", "house"] labels = ["NNP", "NNP", "POS",
    * "NN"]
    *
    * # bert_tokens == ["[CLS]", "john", "johan", "##son", "'", "s", "house", "[SEP]"] #
    * orig_to_tok_map == [1, 2, 4, 6]
    */
  def wordAndSpanLevelAlignmentWithTokenizer(
      tokenLogits: Array[Array[Float]],
      tokenizedSentences: Seq[TokenizedSentence],
      sentence: (WordpieceTokenizedSentence, Int),
      tags: Map[String, Int]): Seq[Annotation] = {

    val labelsWithScores =
      sentence._1.tokens.zip(tokenLogits).flatMap { case (tokenPiece, scores) =>
        val indexedToken = findIndexedToken(tokenizedSentences, sentence, tokenPiece)
        indexedToken.map { token =>
          val label =
            tags.find(_._2 == scores.zipWithIndex.maxBy(_._1)._2).map(_._1).getOrElse("NA")
          val meta = scores.zipWithIndex.flatMap(x =>
            Map(tags.find(_._2 == x._2).map(_._1).getOrElse("NA") -> x._1.toString))
          Annotation(
            annotatorType = AnnotatorType.NAMED_ENTITY,
            begin = token.begin,
            end = token.end,
            result = label,
            metadata = Map("sentence" -> sentence._2.toString, "word" -> token.token) ++ meta)
        }
      }
    labelsWithScores.toSeq
  }

  def findIndexedToken(
      tokenizedSentences: Seq[TokenizedSentence],
      sentence: (WordpieceTokenizedSentence, Int),
      tokenPiece: TokenPiece): Option[IndexedToken]

}

object MergeTokenStrategy {

  val vocab = "vocab"
  val sentencePiece = "sp"

}
