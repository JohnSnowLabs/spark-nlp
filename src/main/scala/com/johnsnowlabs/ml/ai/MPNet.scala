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

import ai.onnxruntime.OnnxTensor
import com.johnsnowlabs.ml.onnx.OnnxWrapper
import com.johnsnowlabs.ml.tensorflow.sign.{ModelSignatureConstants, ModelSignatureManager}
import com.johnsnowlabs.ml.tensorflow.{TensorResources, TensorflowWrapper}
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.ml.util.{ONNX, TensorFlow}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}

import scala.collection.JavaConverters._

/** MPNET Sentence embeddings model
  *
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
private[johnsnowlabs] class MPNet(
    val tensorflowWrapper: Option[TensorflowWrapper],
    val onnxWrapper: Option[OnnxWrapper],
    configProtoBytes: Option[Array[Byte]] = None,
    sentenceStartTokenId: Int,
    sentenceEndTokenId: Int,
    signatures: Option[Map[String, String]] = None)
    extends Serializable {

  private val _tfInstructorSignatures: Map[String, String] =
    signatures.getOrElse(ModelSignatureManager.apply())
  private val paddingTokenId = 1

  val detectedEngine: String =
    if (tensorflowWrapper.isDefined) TensorFlow.name
    else if (onnxWrapper.isDefined) ONNX.name
    else TensorFlow.name

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
        getSentenceEmbeddingFromOnnx(paddedBatch)
      case _ =>
        getSentenceEmbeddingFromTF(paddedBatch)
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

  /** Get sentence embeddings for a batch of sentences
    * @param batch
    *   batch of sentences
    * @return
    *   sentence embeddings
    */
  private def getSentenceEmbeddingFromTF(batch: Seq[Array[Int]]): Array[Array[Float]] = {
    // get max sentence length
    val sequencesLength = batch.map(x => x.length).toArray
    val maxSentenceLength = sequencesLength.max
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

      // pad with padding token
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

  private def getSentenceEmbeddingFromOnnx(batch: Seq[Array[Int]]): Array[Array[Float]] = {
    val batchLength = batch.length
    val maxSentenceLength = batch.map(pieceIds => pieceIds.length).max

    val (runner, env) = onnxWrapper.get.getSession()
    val tokenTensors =
      OnnxTensor.createTensor(env, batch.map(x => x.map(x => x.toLong)).toArray)
    val maskTensors =
      OnnxTensor.createTensor(
        env,
        batch.map(sentence => sentence.map(x => if (x == 0L) 0L else 1L)).toArray)

    val segmentTensors =
      OnnxTensor.createTensor(env, batch.map(x => Array.fill(maxSentenceLength)(0L)).toArray)

    val inputs =
      Map("input_ids" -> tokenTensors, "attention_mask" -> maskTensors).asJava

    // TODO:  A try without a catch or finally is equivalent to putting its body in a block; no exceptions are handled.
    try {
      val results = runner.run(inputs)
      try {
        val embeddings = results
          .get("last_hidden_state")
          .get()
          .asInstanceOf[OnnxTensor]
          .getFloatBuffer
          .array()
        tokenTensors.close()
        maskTensors.close()
        segmentTensors.close()

        val dim = embeddings.length / batchLength
        // group embeddings
        val sentenceEmbeddingsFloatsArray = embeddings.grouped(dim).toArray
        sentenceEmbeddingsFloatsArray
      } finally if (results != null) results.close()
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
        val tokensBatch = batch.map(x => (x._1._1.tokens))
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
