/*
 * Copyright 2017 - 2023  John Snow Labs
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package com.johnsnowlabs.ml.ai

import ai.onnxruntime.{OnnxTensor, TensorInfo}
import com.johnsnowlabs.ml.onnx.{OnnxSession, OnnxWrapper}
import com.johnsnowlabs.ml.openvino.OpenvinoWrapper
import com.johnsnowlabs.ml.tensorflow.sentencepiece.{SentencePieceWrapper, SentencepieceEncoder}
import com.johnsnowlabs.ml.util.{ONNX, Openvino}
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.JavaConverters._
import scala.collection.immutable.ListMap
import scala.collection.mutable

/** BGE-M3 embeddings model (dense + sparse/lexical).
  *
  * BGE-M3 shares the xlm-roberta-large backbone with the existing dense BGE models, but the
  * checkpoint additionally ships a small `sparse_linear` head that produces a per-token lexical
  * importance weight. This class expects an ONNX/OpenVINO graph that has both the dense pooling
  * and the sparse head folded in at export time, exposing two outputs:
  *   - `dense_embedding` of shape `[batch, dim]`, already CLS-pooled and L2-normalized, matching
  *     `BGEM3FlagModel.encode(..., return_dense=True)`
  *   - `token_weights` of shape `[batch, seq]`, already passed through `relu`, used to compute
  *     the sparse lexical weights
  *
  * The sparse weights are remapped from sub-word pieces to token-level weights (max weight per
  * token id, special tokens dropped, only positive weights kept), matching `BGEM3FlagModel`'s
  * lexical weights and the `convert_id_to_token` helper from the model card.
  *
  * @param onnxWrapper
  *   BGE-M3 model wrapper with ONNX Wrapper
  * @param openvinoWrapper
  *   BGE-M3 model wrapper with OpenVINO Wrapper
  * @param spp
  *   XLM-RoBERTa SentencePiece model with SentencePieceWrapper
  * @param caseSensitive
  *   Whether the tokenizer should be case sensitive
  */
private[johnsnowlabs] class BGEM3(
    val onnxWrapper: Option[OnnxWrapper],
    val openvinoWrapper: Option[OpenvinoWrapper],
    val spp: SentencePieceWrapper,
    caseSensitive: Boolean = false)
    extends Serializable {

  import BGEM3._

  protected val logger: Logger = LoggerFactory.getLogger("BGEM3")

  private val DenseOutput = "dense_embedding"
  private val SparseOutput = "token_weights"

  // Model input ids are offset by 1 from the raw SentencePiece ids (XLM-RoBERTa fairseq offset)
  private val pieceIdOffset = 1
  private val SentencePieceDelimiterId = spp.getSppModel.pieceToId("▁")

  val detectedEngine: String =
    if (onnxWrapper.isDefined) ONNX.name
    else if (openvinoWrapper.isDefined) Openvino.name
    else ONNX.name

  private val onnxSessionOptions: Map[String, String] = new OnnxSession().getSessionOptions

  /** Tokenize the input documents with the XLM-RoBERTa SentencePiece model.
    *
    * `maxSentenceLength` bounds the character-level pre-truncation passed to `encodeSentence`,
    * matching the pattern used by every other SentencePiece-based annotator in this codebase
    * (e.g. `XlmRoberta.tokenizeSentence`). Without it, the whole document gets
    * SentencePiece-tokenized before `predict()` truncates the piece-id array down to
    * `maxSentenceLength`, which is wasted work on long documents (BGE-M3 is explicitly meant to
    * handle up to 8192 tokens).
    */
  def tokenize(
      sentences: Seq[Annotation],
      maxSentenceLength: Int): Seq[WordpieceTokenizedSentence] = {
    val encoder =
      new SentencepieceEncoder(spp, caseSensitive, SentencePieceDelimiterId, pieceIdOffset)
    sentences.map { annotation =>
      val sentence = Sentence(
        content = annotation.result,
        start = annotation.begin,
        end = annotation.end,
        metadata = Some(annotation.metadata),
        index = annotation.begin)

      val pieces = encoder.encodeSentence(sentence, maxLength = maxSentenceLength)
      WordpieceTokenizedSentence(pieces)
    }
  }

  /** Predict dense (and optionally sparse) embeddings for a batch of documents.
    *
    * @param sentences
    *   Input annotations (one per document)
    * @param tokenizedSentences
    *   Tokenized documents
    * @param batchSize
    *   Batch size
    * @param maxSentenceLength
    *   Maximum number of tokens per document (including the two special tokens)
    * @param returnSparse
    *   Whether to compute the sparse lexical weights and pack them into the metadata
    * @return
    *   One SENTENCE_EMBEDDINGS annotation per document
    */
  def predict(
      sentences: Seq[Annotation],
      tokenizedSentences: Seq[WordpieceTokenizedSentence],
      batchSize: Int,
      maxSentenceLength: Int,
      returnSparse: Boolean): Seq[Annotation] = {

    tokenizedSentences
      .zip(sentences)
      .grouped(batchSize)
      .toArray
      .flatMap { batch =>
        val tokensBatch: Array[Array[Int]] = batch.map { case (wpSentence, _) =>
          Array(SentenceStartTokenId) ++ wpSentence.tokens
            .map(_.pieceId)
            .take(maxSentenceLength - 2) ++ Array(SentenceEndTokenId)
        }.toArray

        val (denseEmbeddings, sparseWeightsOpt) = getEmbeddings(tokensBatch, returnSparse)

        batch.zipWithIndex.map { case ((_, sentence), idx) =>
          val metadata = sparseWeightsOpt match {
            case Some(sparseWeights) =>
              ListMap(
                (sentence.metadata.toSeq ++ sparseLexicalWeights(
                  tokensBatch(idx),
                  sparseWeights(idx))): _*)
            case None => sentence.metadata
          }

          Annotation(
            annotatorType = AnnotatorType.SENTENCE_EMBEDDINGS,
            begin = sentence.begin,
            end = sentence.end,
            result = sentence.result,
            metadata = metadata,
            embeddings = denseEmbeddings(idx))
        }
      }
  }

  /** Run the encoder and return the dense embeddings and (optionally) the raw per-position sparse
    * weights.
    */
  private def getEmbeddings(
      batch: Array[Array[Int]],
      returnSparse: Boolean): (Array[Array[Float]], Option[Array[Array[Float]]]) = {
    detectedEngine match {
      case Openvino.name => getEmbeddingsOv(batch, returnSparse)
      case _ => getEmbeddingsOnnx(batch, returnSparse)
    }
  }

  private def padBatch(batch: Array[Array[Int]]): (Array[Array[Int]], Array[Array[Long]]) = {
    val maxLength = batch.map(_.length).max
    val padded =
      batch.map(arr => arr ++ Array.fill(maxLength - arr.length)(SentencePadTokenId))
    val attentionMask =
      padded.map(sentence => sentence.map(id => if (id == SentencePadTokenId) 0L else 1L))
    (padded, attentionMask)
  }

  private def getEmbeddingsOnnx(
      batch: Array[Array[Int]],
      returnSparse: Boolean): (Array[Array[Float]], Option[Array[Array[Float]]]) = {
    val (padded, attentionMask) = padBatch(batch)
    val inputIds = padded.map(_.map(_.toLong))

    val (runner, env) = onnxWrapper.get.getSession(onnxSessionOptions)
    val tokenTensors = OnnxTensor.createTensor(env, inputIds)
    val maskTensors = OnnxTensor.createTensor(env, attentionMask)
    val inputs = Map("input_ids" -> tokenTensors, "attention_mask" -> maskTensors).asJava

    try {
      val results = runner.run(inputs)
      try {
        // dense_embedding is already CLS-pooled and L2-normalized in the graph: [batch, dim]
        val denseTensor = results.get(DenseOutput).get().asInstanceOf[OnnxTensor]
        val denseDim = denseTensor.getInfo.getShape.last.toInt
        val dense = denseTensor.getFloatBuffer.array().grouped(denseDim).toArray

        val sparse = if (returnSparse) {
          val sparseOutput = results.get(SparseOutput)
          if (!sparseOutput.isPresent)
            throw new IllegalStateException(
              s"The loaded BGE-M3 ONNX model does not expose a '$SparseOutput' output. " +
                "Re-export the model with the sparse head to use setReturnSparseEmbeddings(true).")
          val flatSparse = sparseOutput.get().asInstanceOf[OnnxTensor].getFloatBuffer.array()
          val width = sparseRowWidth(flatSparse.length, batch.length, padded.head.length)
          Some(flatSparse.grouped(width).toArray)
        } else None

        (dense, sparse)
      } finally {
        if (results != null) results.close()
      }
    } finally {
      tokenTensors.close()
      maskTensors.close()
    }
  }

  private def getEmbeddingsOv(
      batch: Array[Array[Int]],
      returnSparse: Boolean): (Array[Array[Float]], Option[Array[Array[Float]]]) = {
    val (padded, attentionMask) = padBatch(batch)
    val shape = Array(batch.length, padded.head.length)

    val tokenTensors =
      new org.intel.openvino.Tensor(shape, padded.flatMap(_.map(_.toLong)))
    val maskTensors = new org.intel.openvino.Tensor(shape, attentionMask.flatten)

    val inferRequest = openvinoWrapper.get.getCompiledModel().create_infer_request()
    inferRequest.set_tensor("input_ids", tokenTensors)
    inferRequest.set_tensor("attention_mask", maskTensors)
    inferRequest.infer()

    // dense_embedding is already CLS-pooled and L2-normalized in the graph: [batch, dim]
    val denseTensor = inferRequest.get_tensor(DenseOutput)
    val denseDim = denseTensor.get_shape().last
    val dense = denseTensor.data().grouped(denseDim).toArray

    val sparse = if (returnSparse) {
      val flatSparse = inferRequest.get_tensor(SparseOutput).data()
      val width = sparseRowWidth(flatSparse.length, batch.length, padded.head.length)
      Some(flatSparse.grouped(width).toArray)
    } else None

    (dense, sparse)
  }

  /** Validate that the sparse output's flat length matches `[batch, seqLen]` exactly before using
    * it to `grouped(width)` the flat array back into rows. A mismatched shape (e.g. from a bad
    * export) must fail loudly here rather than silently truncating a row's worth of weights via
    * `flatSparse.length / batch.length` integer division, which would misassign weights across
    * the batch without any error.
    */
  private def sparseRowWidth(flatLength: Int, batchSize: Int, seqLen: Int): Int =
    BGEM3.expectedSparseWidth(flatLength, batchSize, seqLen, SparseOutput)

  /** Remap per-position sparse weights to token-level lexical weights, in order of first token
    * occurrence.
    *
    * Follows `BGEM3FlagModel._process_token_weights`: keep only positive weights, drop special
    * tokens, and take the maximum weight for each token id. Token ids are then converted back to
    * their SentencePiece string (`convert_id_to_token`) to form the `{token: weight}` pairs.
    */
  private def sparseLexicalWeights(
      tokens: Array[Int],
      weights: Array[Float]): Seq[(String, String)] = {
    aggregateSparseWeights(tokens, weights).map { case (tokenId, weight) =>
      idToPiece(tokenId) -> weight.toString
    }
  }

  /** Convert a model token id back to its SentencePiece string, inverting the fairseq offset used
    * during encoding (`_convert_id_to_token` in the HF XLM-RoBERTa tokenizer).
    */
  private def idToPiece(tokenId: Int): String = {
    val rawId = tokenId - pieceIdOffset
    if (rawId >= 0) spp.getSppModel.idToPiece(rawId) else tokenId.toString
  }

}

private[johnsnowlabs] object BGEM3 {

  private[ai] val SentenceStartTokenId = 0 // <s>
  private[ai] val SentencePadTokenId = 1 // <pad>
  private[ai] val SentenceEndTokenId = 2 // </s>
  private[ai] val SentenceUnkTokenId = 3 // <unk>

  /** Token ids that never contribute to the sparse lexical weights. */
  private[ai] val unusedTokenIds: Set[Int] =
    Set(SentenceStartTokenId, SentencePadTokenId, SentenceEndTokenId, SentenceUnkTokenId)

  /** Aggregate per-position sparse weights into token-level lexical weights, in order of first
    * token occurrence. Pure and dependency-free (no `spp`/instance state) so it's directly
    * unit-testable.
    *
    * Follows `BGEM3FlagModel._process_token_weights`: keep only positive weights, drop special
    * tokens, and take the maximum weight for each token id.
    */
  private[ai] def aggregateSparseWeights(
      tokens: Array[Int],
      weights: Array[Float]): Seq[(Int, Float)] = {
    val aggregated = mutable.LinkedHashMap.empty[Int, Float]
    val n = math.min(tokens.length, weights.length)
    var i = 0
    while (i < n) {
      val tokenId = tokens(i)
      val weight = weights(i)
      if (weight > 0f && !unusedTokenIds.contains(tokenId)) {
        val previous = aggregated.getOrElse(tokenId, 0f)
        if (weight > previous) aggregated.update(tokenId, weight)
      }
      i += 1
    }
    aggregated.toSeq
  }

  /** Validate that a flat sparse-output buffer's length matches `batchSize * seqLen` exactly,
    * returning the per-row width (`seqLen`) on success. Pure and dependency-free so it's directly
    * unit-testable.
    *
    * @throws IllegalStateException
    *   if the flat length doesn't factor into `batchSize * seqLen`, naming the actual vs.
    *   expected shape rather than letting a silent integer-division truncate/misalign rows.
    */
  private[ai] def expectedSparseWidth(
      flatLength: Int,
      batchSize: Int,
      seqLen: Int,
      outputName: String): Int = {
    val expected = batchSize * seqLen
    if (flatLength != expected)
      throw new IllegalStateException(
        s"The loaded BGE-M3 model's '$outputName' output has an unexpected shape: got a flat " +
          s"length of $flatLength elements, expected $expected (batch=$batchSize x seq=$seqLen). " +
          "Re-export the model so the sparse head output matches [batch, seq].")
    seqLen
  }

}
