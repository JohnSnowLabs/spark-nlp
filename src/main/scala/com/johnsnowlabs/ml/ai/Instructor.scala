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
import com.johnsnowlabs.ml.tensorflow.sentencepiece.SentencePieceWrapper
import com.johnsnowlabs.ml.tensorflow.sign.{ModelSignatureConstants, ModelSignatureManager}
import com.johnsnowlabs.ml.tensorflow.{TensorResources, TensorflowWrapper}
import com.johnsnowlabs.ml.util.{LinAlg, ONNX, TensorFlow}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}

import scala.collection.JavaConverters._

/** InstructorEmbeddings provides the functionality to generate embeddings for instruction and
  * task
  * @param tensorflow
  *   tensorflow wrapper
  * @param configProtoBytes
  *   configProtoBytes
  * @param spp
  *   SentencePieceWrapper
  * @param signatures
  *   signatures
  */

private[johnsnowlabs] class Instructor(
    val tensorflowWrapper: Option[TensorflowWrapper],
    val onnxWrapper: Option[OnnxWrapper],
    val spp: SentencePieceWrapper,
    configProtoBytes: Option[Array[Byte]] = None,
    signatures: Option[Map[String, String]] = None)
    extends Serializable {

  private val _tfInstructorSignatures: Map[String, String] =
    signatures.getOrElse(ModelSignatureManager.apply())
  private val paddingTokenId = 0
  private val eosTokenId = 1
  val detectedEngine: String =
    if (tensorflowWrapper.isDefined) TensorFlow.name
    else if (onnxWrapper.isDefined) ONNX.name
    else TensorFlow.name
  private val onnxSessionOptions: Map[String, String] = new OnnxSession().getSessionOptions

  private def getSentenceEmbeddingFromOnnx(
      batch: Seq[Array[Int]],
      contextLengths: Seq[Int],
      maxSentenceLength: Int): Array[Array[Float]] = {

    val inputIds = batch.map(x => x.map(x => x.toLong)).toArray
    val attentionMask = batch
      .map(sentence => sentence.map(x => if (x == this.paddingTokenId) 0L else 1L))
      .toArray

    val contextMask = attentionMask.zipWithIndex.map { case (batchElement, idx) =>
      batchElement.zipWithIndex.map { case (x, i) =>
        if (i < contextLengths(idx)) 0L else x
      }
    }.toArray

    val (runner, env) = onnxWrapper.get.getSession(onnxSessionOptions)

    val tokenTensors = OnnxTensor.createTensor(env, inputIds)
    val maskTensors = OnnxTensor.createTensor(env, attentionMask)
    val contextTensor =
      OnnxTensor.createTensor(env, contextMask)
    val inputs =
      Map("input_ids" -> tokenTensors, "attention_mask" -> maskTensors).asJava

    // TODO:  A try without a catch or finally is equivalent to putting its body in a block; no exceptions are handled.
    try {
      val results = runner.run(inputs)
      val lastHiddenState = results.get("token_embeddings").get()
      val info = lastHiddenState.getInfo.asInstanceOf[TensorInfo]
      val shape = info.getShape
      try {
        val flattenEmbeddings = lastHiddenState
          .asInstanceOf[OnnxTensor]
          .getFloatBuffer
          .array()
        val embeddings = LinAlg.avgPooling(flattenEmbeddings, contextMask, shape)
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
      contextTensor.close()
    }
  }

  private def padArrayWithZeros(arr: Array[Int], maxLength: Int): Array[Int] = {
    if (arr.length >= maxLength) {
      arr
    } else {
      arr ++ Array.fill(maxLength - arr.length)(0)
    }
  }

  private def getSentenceEmbeddingFromTF(
      paddedBatch: Seq[Array[Int]],
      contextLengths: Seq[Int],
      maxSentenceLength: Int) = {
    // encode batch
    val tensorEncoder = new TensorResources()
    val inputDim = paddedBatch.length * maxSentenceLength
    val batchLength = paddedBatch.length

    // create buffers
    val encoderInputBuffers = tensorEncoder.createIntBuffer(inputDim)
    val encoderAttentionMaskBuffers = tensorEncoder.createIntBuffer(inputDim)
    val encoderContextMaskBuffers = tensorEncoder.createIntBuffer(inputDim)

    val shape = Array(paddedBatch.length.toLong, maxSentenceLength)

    paddedBatch.zipWithIndex.foreach { case (tokenIds, idx) =>
      val offset = idx * maxSentenceLength
      encoderInputBuffers.offset(offset).write(tokenIds)

      // create attention mask
      val mask = tokenIds.map(x => if (x != this.paddingTokenId) 1 else 0)
      encoderAttentionMaskBuffers.offset(offset).write(mask)

      // create context mask
      val contextMask = mask.zipWithIndex.map {
        case (x, i) => { if (i < contextLengths(idx)) 0 else x }
      }
      encoderContextMaskBuffers.offset(offset).write(contextMask)
    }

    // create tensors
    val encoderInputTensors = tensorEncoder.createIntBufferTensor(shape, encoderInputBuffers)
    val encoderAttentionMaskTensors =
      tensorEncoder.createIntBufferTensor(shape, encoderAttentionMaskBuffers)
    val encoderContextMaskTensors =
      tensorEncoder.createIntBufferTensor(shape, encoderContextMaskBuffers)

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
      .feed(
        _tfInstructorSignatures.getOrElse(
          ModelSignatureConstants.EncoderContextMask.key,
          "missing_encoder_context_mask"),
        encoderContextMaskTensors)
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
    encoderContextMaskTensors.close()
    tensorEncoder.clearTensors()
    tensorEncoder.clearSession(sentenceEmbeddings)

    sentenceEmbeddingsFloatsArray

  }

  /** Get sentence embeddings for a batch of sentences
    * @param batch
    *   batch of sentences
    * @param contextLengths
    *   context lengths
    * @return
    *   sentence embeddings
    */
  private def getSentenceEmbedding(
      batch: Seq[Array[Int]],
      contextLengths: Seq[Int]): Array[Array[Float]] = {
    val maxSentenceLength = batch.map(pieceIds => pieceIds.length).max
    val paddedBatch = batch.map(arr => padArrayWithZeros(arr, maxSentenceLength))
    val sentenceEmbeddings: Array[Array[Float]] = detectedEngine match {
      case ONNX.name =>
        getSentenceEmbeddingFromOnnx(paddedBatch, contextLengths, maxSentenceLength)
      case _ => // TF Case
        getSentenceEmbeddingFromTF(paddedBatch, contextLengths, maxSentenceLength)
    }

    sentenceEmbeddings

  }

  /** Tokenize sentences
    * @param sentences
    *   sentences
    * @param task
    *   task
    * @param maxSentenceLength
    *   max sentence length
    * @return
    */
  def tokenize(
      sentences: Seq[Annotation],
      task: String,
      maxSentenceLength: Int): Seq[Array[Int]] = {
    sentences.map(s => {
      val sentWithTask = if (task.nonEmpty) task.concat("").concat(s.result) else s.result
      spp.getSppModel.encodeAsIds(sentWithTask).take(maxSentenceLength - 1) ++ Array(
        this.eosTokenId)
    })
  }

  /** Predict sentence embeddings
    * @param sentences
    *   sentences
    * @param batchSize
    *   batch size
    * @param maxSentenceLength
    *   max sentence length
    * @param instruction
    *   instruction
    * @return
    */
  def predict(
      sentences: Seq[Annotation],
      batchSize: Int,
      maxSentenceLength: Int,
      instruction: String): Seq[Annotation] = {

    val instructionTokenized = spp.getSppModel.encodeAsIds(instruction)
    // repeat instruction length for each sentence
    val instructionTokenizedRepeated: Array[Int] =
      Array.fill(sentences.length)(instructionTokenized.length)

    val batchEmbeddings = sentences.grouped(batchSize).toArray.flatMap { batch =>
      // encode batch
      val batchSP = tokenize(batch, instruction, maxSentenceLength)
      // get sentence embeddings
      val sentenceEmbeddings = getSentenceEmbedding(batchSP, instructionTokenizedRepeated)

      // create annotations
      batch.zip(sentenceEmbeddings).map { case (sentence, vectors) =>
        Annotation(
          annotatorType = AnnotatorType.SENTENCE_EMBEDDINGS,
          begin = sentence.begin,
          end = sentence.end,
          result = sentence.result,
          metadata = sentence.metadata,
          embeddings = vectors)
      }
    }
    batchEmbeddings
  }

}
