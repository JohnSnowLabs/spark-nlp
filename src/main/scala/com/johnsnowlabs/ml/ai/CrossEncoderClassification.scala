/*
 * Copyright 2017-2024 John Snow Labs
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

import ai.onnxruntime.OnnxTensor
import com.johnsnowlabs.ml.onnx.{OnnxSession, OnnxWrapper}
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.annotators.tokenizer.wordpiece.{BasicTokenizer, WordpieceEncoder}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.JavaConverters._

private[johnsnowlabs] class CrossEncoderClassification(
    val onnxWrapper: OnnxWrapper,
    val sentenceStartTokenId: Int,
    val sentenceEndTokenId: Int,
    vocabulary: Map[String, Int])
    extends Serializable {

  protected val logger: Logger = LoggerFactory.getLogger("CrossEncoderClassification")
  protected val sentencePadTokenId = 0

  private val onnxSessionOptions: Map[String, String] = new OnnxSession().getSessionOptions

  private case class EncodedPair(inputIds: Array[Int], tokenTypeIds: Array[Int])

  private def tokenizeDocument(doc: Annotation, caseSensitive: Boolean): Array[Int] = {
    val basicTokenizer = new BasicTokenizer(caseSensitive = caseSensitive, hasBeginEnd = false)
    val encoder = new WordpieceEncoder(vocabulary)
    val sentence = Sentence(doc.result, doc.begin, doc.end, 0)

    val tokens = basicTokenizer.tokenize(sentence)
    val wordpieceTokens =
      if (caseSensitive) tokens.flatMap(token => encoder.encode(token))
      else
        tokens
          .map(x => IndexedToken(x.token.toLowerCase(), x.begin, x.end))
          .flatMap(token => encoder.encode(token))

    wordpieceTokens.map(_.pieceId)
  }

  private def truncatePair(
      seqA: Array[Int],
      seqB: Array[Int],
      maxAvailable: Int): (Array[Int], Array[Int]) = {
    if (seqA.length + seqB.length <= maxAvailable) return (seqA, seqB)

    var lenA = seqA.length
    var lenB = seqB.length
    while (lenA + lenB > maxAvailable) {
      if (lenA > lenB) lenA -= 1 else lenB -= 1
    }
    (seqA.take(lenA), seqB.take(lenB))
  }

  private def encodePair(
      docA: Annotation,
      docB: Annotation,
      caseSensitive: Boolean): EncodedPair = {
    val seqA = tokenizeDocument(docA, caseSensitive)
    val seqB = tokenizeDocument(docB, caseSensitive)

    val maxAvailable = math.max(CrossEncoderClassification.MaxSequenceLength - 3, 0)
    val (truncatedA, truncatedB) = truncatePair(seqA, seqB, maxAvailable)

    val inputIds =
      Array(sentenceStartTokenId) ++ truncatedA ++ Array(sentenceEndTokenId) ++
        truncatedB ++ Array(sentenceEndTokenId)

    val tokenTypeIds =
      Array.fill(1 + truncatedA.length + 1)(0) ++ Array.fill(truncatedB.length + 1)(1)

    EncodedPair(inputIds, tokenTypeIds)
  }

  private def padBatch(batch: Seq[EncodedPair]): (Seq[Array[Int]], Seq[Array[Int]]) = {
    val maxLen = batch.map(_.inputIds.length).max
    val paddedIds =
      batch.map(p => p.inputIds ++ Array.fill(maxLen - p.inputIds.length)(sentencePadTokenId))
    val paddedTypes =
      batch.map(p => p.tokenTypeIds ++ Array.fill(maxLen - p.tokenTypeIds.length)(0))
    (paddedIds, paddedTypes)
  }

  private def computeLogits(
      inputIds: Seq[Array[Int]],
      tokenTypeIds: Seq[Array[Int]]): Array[Array[Float]] = {
    val batchLength = inputIds.length
    val rawScores = computeLogitsWithOnnx(inputIds, tokenTypeIds)
    val dim = rawScores.length / batchLength
    rawScores.grouped(dim).toArray
  }

  private def computeLogitsWithOnnx(
      inputIds: Seq[Array[Int]],
      tokenTypeIds: Seq[Array[Int]]): Array[Float] = {
    val (runner, env) = onnxWrapper.getSession(onnxSessionOptions)

    val tokenTensors = OnnxTensor.createTensor(env, inputIds.map(_.map(_.toLong)).toArray)
    val maskTensors = OnnxTensor.createTensor(
      env,
      inputIds.map(sentence => sentence.map(x => if (x == 0L) 0L else 1L)).toArray)
    val segmentTensors =
      OnnxTensor.createTensor(env, tokenTypeIds.map(_.map(_.toLong)).toArray)

    val inputs =
      Map(
        "input_ids" -> tokenTensors,
        "attention_mask" -> maskTensors,
        "token_type_ids" -> segmentTensors).asJava

    try {
      val results = runner.run(inputs)
      try {
        results.get("logits").get().asInstanceOf[OnnxTensor].getFloatBuffer.array()
      } finally if (results != null) results.close()
    } catch {
      case e: Exception =>
        logger.error("Exception in CrossEncoderClassification ONNX inference", e)
        throw e
    } finally {
      tokenTensors.close()
      maskTensors.close()
      segmentTensors.close()
    }
  }

  def predictScore(
      pairs: Seq[(Annotation, Annotation)],
      batchSize: Int,
      caseSensitive: Boolean): Seq[Annotation] = {
    pairs
      .grouped(batchSize)
      .flatMap { batch =>
        val encoded = batch.map { case (a, b) => encodePair(a, b, caseSensitive) }
        val (paddedIds, paddedTypes) = padBatch(encoded)
        val logits = computeLogits(paddedIds, paddedTypes)

        batch.zip(logits).map { case (_, rowLogits) =>
          val score = sigmoid(rowLogits.head).toString
          Annotation(
            annotatorType = AnnotatorType.CATEGORY,
            begin = 0,
            end = if (score.isEmpty) 0 else score.length - 1,
            result = score,
            metadata = Map("sentence" -> "0", "score" -> score))
        }
      }
      .toSeq
  }

  private def sigmoid(x: Float): Float = (1.0 / (1.0 + math.exp(-x))).toFloat

}

private[johnsnowlabs] object CrossEncoderClassification {
  val MaxSequenceLength = 512
}
