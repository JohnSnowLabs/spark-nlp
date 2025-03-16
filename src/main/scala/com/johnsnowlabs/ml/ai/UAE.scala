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
import com.johnsnowlabs.ml.tensorflow.{TensorResources, TensorflowWrapper}
import com.johnsnowlabs.ml.tensorflow.sign.{ModelSignatureConstants, ModelSignatureManager}
import com.johnsnowlabs.ml.util.{LinAlg, ONNX, Openvino, TensorFlow}
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}

import scala.collection.JavaConverters._
import scala.util.Try

/** UAE Sentence embeddings model
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
private[johnsnowlabs] class UAE(
    val tensorflowWrapper: Option[TensorflowWrapper],
    val onnxWrapper: Option[OnnxWrapper],
    val openvinoWrapper: Option[OpenvinoWrapper],
    configProtoBytes: Option[Array[Byte]] = None,
    sentenceStartTokenId: Int,
    sentenceEndTokenId: Int,
    signatures: Option[Map[String, String]] = None) {

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
    *
    * @param batch
    *   batch of sentences
    * @return
    *   sentence embeddings
    */
  private def getSentenceEmbedding(
      batch: Seq[Array[Int]],
      poolingStrategy: String): Array[Array[Float]] = {
    val maxSentenceLength = batch.map(pieceIds => pieceIds.length).max
    val paddedBatch = batch.map(arr => padArrayWithZeros(arr, maxSentenceLength))
    val sentenceEmbeddings: Array[Array[Float]] = detectedEngine match {
      case ONNX.name =>
        getSentenceEmbeddingFromOnnx(paddedBatch, maxSentenceLength, poolingStrategy)
      case Openvino.name =>
        getSentenceEmbeddingFromOpenvino(paddedBatch, maxSentenceLength, poolingStrategy)
      case _ => // TF Case
        getSentenceEmbeddingFromTF(paddedBatch, maxSentenceLength, poolingStrategy)
    }

    sentenceEmbeddings
  }

  /** Pools word embeddings to sentence embeddings given a strategy.
    *
    * @param embeddings
    *   A 3D array of Floats representing the embeddings. The dimensions are [batch_size,
    *   sequence_length, embedding_dim].
    * @param attentionMask
    *   A 2D array of Longs representing the attention mask. The dimensions are [batch_size,
    *   sequence_length].
    * @param poolingStrategy
    *   A String representing the pooling strategy to be applied. The following strategies are
    *   supported:
    *
    *   - `"cls"`: leading `[CLS]` token
    *   - `"cls_avg"`: leading `[CLS]` token + mean of all other tokens
    *   - `"last"`: embeddings of the last token in the sequence
    *   - `"avg"`: mean of all tokens
    *   - `"max"`: max of all embedding values for the token sequence
    *   - `"int"`: An integer number, which represents the index of the token to use as the
    *     embedding
    * @return
    *   A 2D array of Floats representing the pooled embeddings. The dimensions are [batch_size,
    *   embedding_dim].
    */
  private def pool(
      embeddings: Array[Array[Array[Float]]],
      attentionMask: Array[Array[Long]],
      poolingStrategy: String): Array[Array[Float]] = {
    poolingStrategy match {
      case "cls" => LinAlg.clsPooling(embeddings, attentionMask)
      case "cls_avg" => LinAlg.clsAvgPooling(embeddings, attentionMask)
      case "last" => LinAlg.lastPooling(embeddings, attentionMask)
      case "avg" =>
        val shape: Array[Long] =
          Array(embeddings.length, embeddings.head.length, embeddings.head.head.length)
        val avgPooled = LinAlg.avgPooling(embeddings.flatten.flatten, attentionMask, shape)
        avgPooled.t.toArray.grouped(avgPooled.cols).toArray
      case "max" => LinAlg.maxPooling(embeddings, attentionMask)
      case index if Try(index.toInt).isSuccess => LinAlg.tokenPooling(embeddings, index.toInt)
      case _ =>
        throw new IllegalArgumentException(s"Pooling strategy $poolingStrategy not supported.")
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
      batch: Seq[Array[Int]],
      maxSentenceLength: Int,
      poolingStrategy: String): Array[Array[Float]] = {
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
    val embeddingDim = sentenceEmbeddingsFloats.length / maxSentenceLength / batchLength

    // group embeddings
    val sentenceEmbeddingsFloatsArray =
      sentenceEmbeddingsFloats.grouped(embeddingDim).toArray.grouped(maxSentenceLength).toArray

    val attentionMask: Array[Array[Long]] =
      TensorResources.extractLongs(encoderAttentionMaskTensors).grouped(maxSentenceLength).toArray

    // close buffers
    sentenceEmbeddings.foreach(_.close())
    encoderInputTensors.close()
    encoderAttentionMaskTensors.close()
    tensorEncoder.clearTensors()
    tensorEncoder.clearSession(sentenceEmbeddings)

    pool(sentenceEmbeddingsFloatsArray, attentionMask, poolingStrategy)
  }

  private def getSentenceEmbeddingFromOpenvino(
      batch: Seq[Array[Int]],
      maxSentenceLength: Int,
      poolingStrategy: String): Array[Array[Float]] = {

    val batchLength = batch.length
    val shape = Array(batchLength, maxSentenceLength)
    val tokenTensors =
      new org.intel.openvino.Tensor(shape, batch.flatMap(x => x.map(xx => xx.toLong)).toArray)

    val attentionMask = batch.map(sentence => sentence.map(x => if (x == 0L) 0L else 1L)).toArray
    val maskTensors = new org.intel.openvino.Tensor(shape, attentionMask.flatten)
    val segmentTensors =
      new org.intel.openvino.Tensor(
        shape,
        batch.map(x => Array.fill(maxSentenceLength)(0L)).toArray.flatten)

    val inferRequest = openvinoWrapper.get.getCompiledModel().create_infer_request()
    inferRequest.set_tensor("input_ids", tokenTensors)
    inferRequest.set_tensor("attention_mask", maskTensors)
    inferRequest.set_tensor("token_type_ids", segmentTensors)

    inferRequest.infer()

    val embeddings =
      try {
        val lastHiddenState = inferRequest
          .get_tensor("last_hidden_state")
        val shape = lastHiddenState.get_shape()
        val Array(_, sequenceLength, embeddingDim) = shape
        try {
          val flattenEmbeddings = lastHiddenState.data()

          flattenEmbeddings.grouped(embeddingDim).toArray.grouped(sequenceLength).toArray
        }
      }

    pool(embeddings, attentionMask, poolingStrategy)

  }

  private def getSentenceEmbeddingFromOnnx(
      batch: Seq[Array[Int]],
      maxSentenceLength: Int,
      poolingStrategy: String): Array[Array[Float]] = {

    val inputIds = batch.map(x => x.map(x => x.toLong)).toArray
    val attentionMask = batch.map(sentence => sentence.map(x => if (x == 0L) 0L else 1L)).toArray

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
    val embeddings =
      try {
        val results = runner.run(inputs)
        val lastHiddenState = results.get("last_hidden_state").get()
        val info = lastHiddenState.getInfo.asInstanceOf[TensorInfo]
        val shape = info.getShape.map(_.toInt)
        val Array(_, sequenceLength, embeddingDim) = shape
        try {
          val flattenEmbeddings = lastHiddenState
            .asInstanceOf[OnnxTensor]
            .getFloatBuffer
            .array()
          tokenTensors.close()
          maskTensors.close()
          segmentTensors.close()

          flattenEmbeddings.grouped(embeddingDim).toArray.grouped(sequenceLength).toArray
        } finally if (results != null) results.close()
      }

    pool(embeddings, attentionMask, poolingStrategy)
  }

  /** Predict sentence embeddings for a batch of sentences
    *
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
      maxSentenceLength: Int,
      poolingStrategy: String): Seq[Annotation] = {

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

        val sentenceEmbeddings = getSentenceEmbedding(tokens, poolingStrategy)

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
