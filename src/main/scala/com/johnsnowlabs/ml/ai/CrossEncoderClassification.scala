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
import com.johnsnowlabs.ml.openvino.OpenvinoWrapper
import com.johnsnowlabs.ml.tensorflow.sign.{ModelSignatureConstants, ModelSignatureManager}
import com.johnsnowlabs.ml.tensorflow.{TensorResources, TensorflowWrapper}
import com.johnsnowlabs.ml.util.{ONNX, Openvino, TensorFlow}
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.annotators.tokenizer.wordpiece.{BasicTokenizer, WordpieceEncoder}
import com.johnsnowlabs.nlp.{ActivationFunction, Annotation, AnnotatorType}
import org.intel.openvino.Tensor
import org.slf4j.{Logger, LoggerFactory}
import org.tensorflow.ndarray.buffer.IntDataBuffer

import scala.collection.JavaConverters._

/** Cross-encoder scoring for pairs of documents.
  *
  * Unlike [[BertClassification]], which classifies a single (optionally paired) sequence, this
  * class jointly encodes a pair of texts as a single sequence `[CLS] text_a [SEP] text_b [SEP]`
  * with pair-aware `token_type_ids` and runs a single forward pass through a BERT-family
  * transformer that carries a sequence classification/regression head. One raw logit vector is
  * produced per pair and reduced to a single score, matching the behavior of
  * `sentence-transformers` `CrossEncoder`.
  *
  * @param tensorflowWrapper
  *   Bert Model wrapper with TensorFlow Wrapper
  * @param onnxWrapper
  *   Bert Model wrapper with ONNX Wrapper
  * @param openvinoWrapper
  *   Bert Model wrapper with OpenVINO Wrapper
  * @param sentenceStartTokenId
  *   Id of the sentence start token (`[CLS]`)
  * @param sentenceEndTokenId
  *   Id of the sentence separator token (`[SEP]`)
  * @param tags
  *   labels the classification head was trained with, in order (empty for regression heads)
  * @param signatures
  *   TF v2 signatures in Spark NLP
  * @param vocabulary
  *   WordPiece vocabulary used to encode the words to ids
  */
private[johnsnowlabs] class CrossEncoderClassification(
    val tensorflowWrapper: Option[TensorflowWrapper],
    val onnxWrapper: Option[OnnxWrapper],
    val openvinoWrapper: Option[OpenvinoWrapper],
    val sentenceStartTokenId: Int,
    val sentenceEndTokenId: Int,
    configProtoBytes: Option[Array[Byte]] = None,
    tags: Map[String, Int],
    signatures: Option[Map[String, String]] = None,
    vocabulary: Map[String, Int])
    extends Serializable {

  protected val logger: Logger = LoggerFactory.getLogger("CrossEncoderClassification")
  val _tfSignatures: Map[String, String] = signatures.getOrElse(ModelSignatureManager.apply())

  protected val sentencePadTokenId = 0

  val detectedEngine: String =
    if (tensorflowWrapper.isDefined) TensorFlow.name
    else if (onnxWrapper.isDefined) ONNX.name
    else if (openvinoWrapper.isDefined) Openvino.name
    else TensorFlow.name

  private val onnxSessionOptions: Map[String, String] = new OnnxSession().getSessionOptions

  /** An encoded pair: the joint input ids and their matching token type (segment) ids. */
  private case class EncodedPair(inputIds: Array[Int], tokenTypeIds: Array[Int])

  /** WordPiece-tokenizes a single document to a flat sequence of piece ids. */
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

  /** Pair-aware truncation. BERT-family backbones impose a hard `maxSentenceLength` ceiling that
    * is shared across both texts combined (not per text), so truncation must consider both
    * sequences jointly.
    *
    *   - `longest_first` (HuggingFace default): repeatedly drop the last token of whichever
    *     sequence is currently longer until the pair fits.
    *   - `query_first`: keep the first text (the query) intact as far as possible and give the
    *     remaining budget to the second text.
    *
    * @param seqA
    *   piece ids of the first text
    * @param seqB
    *   piece ids of the second text
    * @param maxAvailable
    *   token budget for both texts combined (already excludes the special tokens)
    * @param truncationStrategy
    *   `"longest_first"` or `"query_first"`
    * @return
    *   the (possibly) truncated pair of piece id sequences
    */
  private def truncatePair(
      seqA: Array[Int],
      seqB: Array[Int],
      maxAvailable: Int,
      truncationStrategy: String): (Array[Int], Array[Int]) = {

    if (seqA.length + seqB.length <= maxAvailable) return (seqA, seqB)

    truncationStrategy match {
      case CrossEncoderClassification.QueryFirst =>
        val truncatedA = seqA.take(maxAvailable)
        val truncatedB = seqB.take(maxAvailable - truncatedA.length)
        (truncatedA, truncatedB)

      case _ => // longest_first
        var lenA = seqA.length
        var lenB = seqB.length
        while (lenA + lenB > maxAvailable) {
          if (lenA > lenB) lenA -= 1 else lenB -= 1
        }
        (seqA.take(lenA), seqB.take(lenB))
    }
  }

  /** Jointly encodes a pair of documents into `[CLS] a [SEP] b [SEP]` with matching token type
    * ids (0 for `[CLS] a [SEP]`, 1 for `b [SEP]`).
    */
  private def encodePair(
      docA: Annotation,
      docB: Annotation,
      maxSentenceLength: Int,
      caseSensitive: Boolean,
      truncationStrategy: String): EncodedPair = {

    val seqA = tokenizeDocument(docA, caseSensitive)
    val seqB = tokenizeDocument(docB, caseSensitive)

    // budget excludes the 3 special tokens: [CLS] ... [SEP] ... [SEP]
    val maxAvailable = math.max(maxSentenceLength - 3, 0)
    val (truncatedA, truncatedB) =
      truncatePair(seqA, seqB, maxAvailable, truncationStrategy)

    val inputIds =
      (Array(sentenceStartTokenId) ++ truncatedA ++ Array(sentenceEndTokenId) ++
        truncatedB ++ Array(sentenceEndTokenId))

    // segment 0 covers [CLS] a [SEP]; segment 1 covers b [SEP]
    val segmentZeros = 1 + truncatedA.length + 1
    val segmentOnes = truncatedB.length + 1
    val tokenTypeIds = Array.fill(segmentZeros)(0) ++ Array.fill(segmentOnes)(1)

    EncodedPair(inputIds, tokenTypeIds)
  }

  /** Right-pads a batch of encoded pairs to a common length so they can be stacked into a single
    * tensor. Padding uses the pad token id for input ids and segment 0 for token type ids;
    * attention masks (derived downstream) keep padded positions inert.
    */
  private def padBatch(batch: Seq[EncodedPair]): (Seq[Array[Int]], Seq[Array[Int]]) = {
    val maxLen = batch.map(_.inputIds.length).max
    val paddedIds = batch.map { p =>
      p.inputIds ++ Array.fill(maxLen - p.inputIds.length)(sentencePadTokenId)
    }
    val paddedTypes = batch.map { p =>
      p.tokenTypeIds ++ Array.fill(maxLen - p.tokenTypeIds.length)(0)
    }
    (paddedIds, paddedTypes)
  }

  /** Runs one forward pass over an already padded batch and returns the raw logits per row. */
  private def computeLogits(
      inputIds: Seq[Array[Int]],
      tokenTypeIds: Seq[Array[Int]]): Array[Array[Float]] = {

    val batchLength = inputIds.length
    val maxSentenceLength = inputIds.head.length

    val rawScores = detectedEngine match {
      case ONNX.name => computeLogitsWithOnnx(inputIds, tokenTypeIds)
      case Openvino.name => computeLogitsWithOv(inputIds, tokenTypeIds, maxSentenceLength)
      case _ => computeLogitsWithTF(inputIds, tokenTypeIds, maxSentenceLength)
    }

    val dim = rawScores.length / batchLength
    rawScores.grouped(dim).toArray
  }

  private def computeLogitsWithOnnx(
      inputIds: Seq[Array[Int]],
      tokenTypeIds: Seq[Array[Int]]): Array[Float] = {

    val (runner, env) = onnxWrapper.get.getSession(onnxSessionOptions)

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
        results
          .get("logits")
          .get()
          .asInstanceOf[OnnxTensor]
          .getFloatBuffer
          .array()
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

  private def computeLogitsWithOv(
      inputIds: Seq[Array[Int]],
      tokenTypeIds: Seq[Array[Int]],
      maxSentenceLength: Int): Array[Float] = {

    val batchLength = inputIds.length
    val shape = Array(batchLength, maxSentenceLength)

    val tokenTensors =
      new Tensor(shape, inputIds.flatMap(_.map(_.toLong)).toArray)
    val maskTensors =
      new Tensor(
        shape,
        inputIds.flatMap(sentence => sentence.map(x => if (x == 0L) 0L else 1L)).toArray)
    val segmentTensors =
      new Tensor(shape, tokenTypeIds.flatMap(_.map(_.toLong)).toArray)

    val inferRequest = openvinoWrapper.get.getCompiledModel().create_infer_request()
    inferRequest.set_tensor("input_ids", tokenTensors)
    inferRequest.set_tensor("attention_mask", maskTensors)
    inferRequest.set_tensor("token_type_ids", segmentTensors)

    inferRequest.infer()

    try {
      inferRequest.get_tensor("logits").data()
    } catch {
      case e: Exception =>
        logger.error("Exception in CrossEncoderClassification OpenVINO inference", e)
        throw e
    }
  }

  private def computeLogitsWithTF(
      inputIds: Seq[Array[Int]],
      tokenTypeIds: Seq[Array[Int]],
      maxSentenceLength: Int): Array[Float] = {

    val batchLength = inputIds.length
    val tensors = new TensorResources()

    val tokenBuffers: IntDataBuffer = tensors.createIntBuffer(batchLength * maxSentenceLength)
    val maskBuffers: IntDataBuffer = tensors.createIntBuffer(batchLength * maxSentenceLength)
    val segmentBuffers: IntDataBuffer = tensors.createIntBuffer(batchLength * maxSentenceLength)

    val shape = Array(batchLength.toLong, maxSentenceLength)

    inputIds.zip(tokenTypeIds).zipWithIndex.foreach { case ((sentence, segments), idx) =>
      val offset = idx * maxSentenceLength
      tokenBuffers.offset(offset).write(sentence)
      maskBuffers.offset(offset).write(sentence.map(x => if (x == 0) 0 else 1))
      segmentBuffers.offset(offset).write(segments)
    }

    val session = tensorflowWrapper.get.getTFSessionWithSignature(
      configProtoBytes = configProtoBytes,
      savedSignatures = signatures,
      initAllTables = false)
    val runner = session.runner

    val tokenTensors = tensors.createIntBufferTensor(shape, tokenBuffers)
    val maskTensors = tensors.createIntBufferTensor(shape, maskBuffers)
    val segmentTensors = tensors.createIntBufferTensor(shape, segmentBuffers)

    runner
      .feed(
        _tfSignatures.getOrElse(ModelSignatureConstants.InputIds.key, "missing_input_id_key"),
        tokenTensors)
      .feed(
        _tfSignatures
          .getOrElse(ModelSignatureConstants.AttentionMask.key, "missing_input_mask_key"),
        maskTensors)
      .feed(
        _tfSignatures
          .getOrElse(ModelSignatureConstants.TokenTypeIds.key, "missing_segment_ids_key"),
        segmentTensors)
      .fetch(
        _tfSignatures.getOrElse(ModelSignatureConstants.LogitsOutput.key, "missing_logits_key"))

    val outs = runner.run().asScala
    val rawScores = TensorResources.extractFloats(outs.head)

    outs.foreach(_.close())
    tensors.clearSession(outs)
    tensors.clearTensors()

    rawScores
  }

  /** Reduces a raw logit vector to a single score, following the `sentence-transformers`
    * convention: regression heads (a single logit) emit the activated logit directly, while
    * multi-label heads emit the activated score of the top class.
    */
  private def scoreFromLogits(
      logits: Array[Float],
      activation: String): (String, Array[Float]) = {
    val activated = activation match {
      case ActivationFunction.sigmoid => logits.map(sigmoid)
      case ActivationFunction.identity => logits
      case _ => softmax(logits)
    }
    (activated.zipWithIndex.maxBy(_._1)._2.toString, activated)
  }

  /** Scores a batch of document pairs. Row `i` of `pairs` yields exactly one score Annotation of
    * type CATEGORY, with no cross-row interaction.
    *
    * @param pairs
    *   row-aligned pairs of documents to jointly score
    * @param batchSize
    *   number of pairs processed per forward pass
    * @param maxSentenceLength
    *   combined sequence length ceiling for both texts (already capped to the model config)
    * @param caseSensitive
    *   whether to keep token casing
    * @param activation
    *   `"sigmoid"`, `"softmax"`, or `"identity"`
    * @param truncationStrategy
    *   `"longest_first"` or `"query_first"`
    * @return
    *   one CATEGORY Annotation per input pair, in input order
    */
  def predictScore(
      pairs: Seq[(Annotation, Annotation)],
      batchSize: Int,
      maxSentenceLength: Int,
      caseSensitive: Boolean,
      activation: String,
      truncationStrategy: String): Seq[Annotation] = {

    val isRegression = tags.size <= 1

    pairs
      .grouped(batchSize)
      .flatMap { batch =>
        val encoded =
          batch.map { case (a, b) =>
            encodePair(a, b, maxSentenceLength, caseSensitive, truncationStrategy)
          }
        val (paddedIds, paddedTypes) = padBatch(encoded)
        val logits = computeLogits(paddedIds, paddedTypes)

        batch.zip(logits).map { case ((_, docB), rowLogits) =>
          val (topIdx, activated) = scoreFromLogits(rowLogits, activation)

          // Regression / single-logit heads: the score itself is the result.
          // Multi-label heads: the top label is the result, per-label scores go to metadata.
          val (result, score) =
            if (isRegression) {
              val s = activated.head
              (s.toString, s)
            } else {
              val label = tags.find(_._2 == topIdx.toInt).map(_._1).getOrElse(topIdx)
              (label, activated(topIdx.toInt))
            }

          val labelScores: Map[String, String] =
            if (isRegression) Map.empty
            else
              activated.zipWithIndex.map { case (s, i) =>
                tags.find(_._2 == i).map(_._1).getOrElse(i.toString) -> s.toString
              }.toMap

          Annotation(
            annotatorType = AnnotatorType.CATEGORY,
            begin = 0,
            end = if (result.isEmpty) 0 else result.length - 1,
            result = result,
            metadata = Map("sentence" -> "0", "score" -> score.toString) ++ labelScores)
        }
      }
      .toSeq
  }

  private def sigmoid(x: Float): Float = (1.0 / (1.0 + math.exp(-x))).toFloat

  private def softmax(scores: Array[Float]): Array[Float] = {
    val maxScore = if (scores.isEmpty) 0.0 else scores.max.toDouble
    val exp = scores.map(x => math.exp(x - maxScore))
    val sum = exp.sum
    exp.map(x => (x / sum).toFloat)
  }

}

private[johnsnowlabs] object CrossEncoderClassification {
  val LongestFirst = "longest_first"
  val QueryFirst = "query_first"
}
