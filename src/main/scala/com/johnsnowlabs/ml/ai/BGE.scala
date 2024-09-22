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
import com.johnsnowlabs.ml.ai.util.PrepareEmbeddings
import com.johnsnowlabs.ml.onnx.{OnnxSession, OnnxWrapper}
import com.johnsnowlabs.ml.openvino.OpenvinoWrapper
import com.johnsnowlabs.ml.tensorflow.sign.{ModelSignatureConstants, ModelSignatureManager}
import com.johnsnowlabs.ml.tensorflow.{TensorResources, TensorflowWrapper}
import com.johnsnowlabs.ml.util.{LinAlg, ONNX, Openvino, TensorFlow}
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import org.intel.openvino.Tensor
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.JavaConverters._

/** BGE Sentence embeddings model
  * @param tensorflowWrapper
  *   tensorflow wrapper
  * @param configProtoBytes
  *   config proto bytes
  * @param sentenceStartTokenId
  *   sentence start token id
  * @param sentenceEndTokenId
  *   sentence end token id
  * @param signatures
  *   signatures
  */
private[johnsnowlabs] class BGE(
    val tensorflowWrapper: Option[TensorflowWrapper],
    val onnxWrapper: Option[OnnxWrapper],
    val openvinoWrapper: Option[OpenvinoWrapper],
    configProtoBytes: Option[Array[Byte]] = None,
    sentenceStartTokenId: Int,
    sentenceEndTokenId: Int,
    signatures: Option[Map[String, String]] = None)
    extends Serializable {

  protected val logger: Logger = LoggerFactory.getLogger("BGE")

  private val _tfInstructorSignatures: Map[String, String] =
    signatures.getOrElse(ModelSignatureManager.apply())
  private val paddingTokenId = 0

  val detectedEngine: String =
    if (tensorflowWrapper.isDefined) TensorFlow.name
    else if (onnxWrapper.isDefined) ONNX.name
    else if (openvinoWrapper.isDefined) Openvino.name
    else TensorFlow.name
  private val onnxSessionOptions: Map[String, String] = new OnnxSession().getSessionOptions

  /** Get sentence embeddings for a batch of sentences
    * @param batch
    *   batch of sentences
    * @return
    *   sentence embeddings
    */
  private def getSentenceEmbedding(batch: Seq[Array[Int]]): Array[Array[Float]] = {
    val maxSentenceLength = batch.map(pieceIds => pieceIds.length).max
    val paddedBatch = batch.map(arr => padArrayWithZeros(arr, maxSentenceLength))
    val embeddings = detectedEngine match {
      case ONNX.name =>
        getSentenceEmbeddingFromOnnx(paddedBatch, maxSentenceLength)

      case Openvino.name =>
        getSentenceEmbeddingFromOv(paddedBatch, maxSentenceLength)
      case _ =>
        getSentenceEmbeddingFromTF(paddedBatch, maxSentenceLength)
    }
    embeddings
  }

  private def padArrayWithZeros(arr: Array[Int], maxLength: Int): Array[Int] = {
    if (arr.length >= maxLength) {
      arr
    } else {
      arr ++ Array.fill(maxLength - arr.length)(0)
    }
  }

  private def getSentenceEmbeddingFromTF(
      batch: Seq[Array[Int]],
      maxSentenceLength: Int): Array[Array[Float]] = {
    val batchLength = batch.length

    // encode batch
    val tensorEncoder = new TensorResources()
    val inputDim = batch.length * maxSentenceLength

    // create buffers
    val encoderInputBuffers = tensorEncoder.createIntBuffer(inputDim)
    val encoderAttentionMaskBuffers = tensorEncoder.createIntBuffer(inputDim)

    val shape = Array(batch.length.toLong, maxSentenceLength)

    batch.zipWithIndex.foreach { case (tokenIds, idx) =>
      val offset = idx * maxSentenceLength
      val diff = maxSentenceLength - tokenIds.length

      // pad with 0
      val s = tokenIds.take(maxSentenceLength) ++ Array.fill[Int](diff)(this.paddingTokenId)
      encoderInputBuffers.offset(offset).write(s)

      // create attention mask
      val mask = s.map(x => if (x != this.paddingTokenId) 1 else 0)
      encoderAttentionMaskBuffers.offset(offset).write(mask)

    }

    // create tensors
    val encoderInputTensors = tensorEncoder.createIntBufferTensor(shape, encoderInputBuffers)
    val encoderAttentionMaskTensors =
      tensorEncoder.createIntBufferTensor(shape, encoderAttentionMaskBuffers)

    // run model
    val runner = tensorflowWrapper.get
      .getTFSessionWithSignature(
        configProtoBytes = configProtoBytes,
        initAllTables = false,
        savedSignatures = signatures)
      .runner

    runner
      .feed(
        _tfInstructorSignatures.getOrElse(
          ModelSignatureConstants.EncoderInputIds.key,
          "missing_encoder_input_ids"),
        encoderInputTensors)
      .feed(
        _tfInstructorSignatures.getOrElse(
          ModelSignatureConstants.EncoderAttentionMask.key,
          "missing_encoder_attention_mask"),
        encoderAttentionMaskTensors)
      .fetch(_tfInstructorSignatures
        .getOrElse(ModelSignatureConstants.LastHiddenState.key, "missing_last_hidden_state"))

    // get embeddings
    val sentenceEmbeddings = runner.run().asScala
    val sentenceEmbeddingsFloats = TensorResources.extractFloats(sentenceEmbeddings.head)
    val dim = sentenceEmbeddingsFloats.length / batchLength

    // group embeddings
    val sentenceEmbeddingsFloatsArray = sentenceEmbeddingsFloats.grouped(dim).toArray

    // close buffers
    sentenceEmbeddings.foreach(_.close())
    encoderInputTensors.close()
    encoderAttentionMaskTensors.close()
    tensorEncoder.clearTensors()
    tensorEncoder.clearSession(sentenceEmbeddings)

    sentenceEmbeddingsFloatsArray
  }



  private def getSentenceEmbeddingFromOv(
                                            batch: Seq[Array[Int]],
                                            maxSentenceLength: Int): Array[Array[Float]] = {


    val batchLength = batch.length
    val shape = Array(batchLength, maxSentenceLength)
    val tokenTensors =
      new org.intel.openvino.Tensor(shape, batch.flatMap(x => x.map(xx => xx.toLong)).toArray)
    val attentionMask = batch.map(sentence => sentence.map(x => if (x < 0L) 0L else 1L)).toArray

    val maskTensors = new org.intel.openvino.Tensor(
      shape,
      attentionMask.flatten)

    val segmentTensors = new Tensor(shape, Array.fill(batchLength * maxSentenceLength)(0L))
    val inferRequest = openvinoWrapper.get.getCompiledModel().create_infer_request()
    inferRequest.set_tensor("input_ids", tokenTensors)
    inferRequest.set_tensor("attention_mask", maskTensors)
    inferRequest.set_tensor("token_type_ids", segmentTensors)

    inferRequest.infer()

    try {
      try {
        val lastHiddenState = inferRequest
          .get_tensor("last_hidden_state")
        val shape = lastHiddenState.get_shape().map(_.toLong)
       val flattenEmbeddings =  lastHiddenState
          .data()
        val embeddings = LinAlg.avgPooling(flattenEmbeddings, attentionMask, shape)
        val normalizedEmbeddings = LinAlg.l2Normalize(embeddings)
        LinAlg.denseMatrixToArray(normalizedEmbeddings)

      }
    } catch {
      case e: Exception =>
        e.printStackTrace()
        Array.empty[Float]
        // Rethrow the exception to propagate it further
        throw e
    }

  }


  private def getSentenceEmbeddingFromOnnx(
      batch: Seq[Array[Int]],
      maxSentenceLength: Int): Array[Array[Float]] = {

    val inputIds = batch.map(x => x.map(x => x.toLong)).toArray
    val attentionMask = batch.map(sentence => sentence.map(x => if (x < 0L) 0L else 1L)).toArray

    val (runner, env) = onnxWrapper.get.getSession(onnxSessionOptions)

    val tokenTensors = OnnxTensor.createTensor(env, inputIds)
    val maskTensors = OnnxTensor.createTensor(env, attentionMask)
    val segmentTensors =
      OnnxTensor.createTensor(env, batch.map(x => Array.fill(maxSentenceLength)(0L)).toArray)
    val inputs =
      Map(
        "input_ids" -> tokenTensors,
        "attention_mask" -> maskTensors,
        "token_type_ids" -> segmentTensors).asJava

    // TODO:  A try without a catch or finally is equivalent to putting its body in a block; no exceptions are handled.
    try {
      val results = runner.run(inputs)
      val lastHiddenState = results.get("last_hidden_state").get()
      val info = lastHiddenState.getInfo.asInstanceOf[TensorInfo]
      val shape = info.getShape
      try {
        val flattenEmbeddings = lastHiddenState
          .asInstanceOf[OnnxTensor]
          .getFloatBuffer
          .array()

        val embeddings = LinAlg.avgPooling(flattenEmbeddings, attentionMask, shape)
        val normalizedEmbeddings = LinAlg.l2Normalize(embeddings)
        LinAlg.denseMatrixToArray(normalizedEmbeddings)
      } finally if (results != null) results.close()
    } catch {
      case e: Exception =>
        // Handle exceptions by logging or other means.
        e.printStackTrace()
        Array.empty[Array[Float]] // Return an empty array or appropriate error handling
    } finally {
      // Close tensors outside the try-catch to avoid repeated null checks.
      // These resources are initialized before the try-catch, so they should be closed here.
      tokenTensors.close()
      maskTensors.close()
      segmentTensors.close()
    }
  }

  /** Predict sentence embeddings for a batch of sentences
    * @param sentences
    *   sentences
    * @param tokenizedSentences
    *   tokenized sentences
    * @param batchSize
    *   batch size
    * @param maxSentenceLength
    *   max sentence length
    * @return
    */
  def predict(
      sentences: Seq[Annotation],
      tokenizedSentences: Seq[WordpieceTokenizedSentence],
      batchSize: Int,
      maxSentenceLength: Int): Seq[Annotation] = {

    tokenizedSentences
      .zip(sentences)
      .zipWithIndex
      .grouped(batchSize)
      .toArray
      .flatMap { batch =>
        val tokensBatch = batch.map(x => x._1._1.tokens)
        val tokens = tokensBatch.map(x =>
          Array(sentenceStartTokenId) ++ x
            .map(y => y.pieceId)
            .take(maxSentenceLength - 2) ++ Array(sentenceEndTokenId))

        val sentenceEmbeddings = getSentenceEmbedding(tokens)

        batch.zip(sentenceEmbeddings).map { case (sentence, vectors) =>
          Annotation(
            annotatorType = AnnotatorType.SENTENCE_EMBEDDINGS,
            begin = sentence._1._2.begin,
            end = sentence._1._2.end,
            result = sentence._1._2.result,
            metadata = sentence._1._2.metadata,
            embeddings = vectors)
        }
      }
  }

}
