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

import ai.onnxruntime.OnnxTensor
import com.johnsnowlabs.ml.ai.util.PrepareEmbeddings
import com.johnsnowlabs.ml.onnx.{OnnxSession, OnnxWrapper}
import com.johnsnowlabs.ml.openvino.OpenvinoWrapper
import com.johnsnowlabs.ml.tensorflow.sign.{ModelSignatureConstants, ModelSignatureManager}
import com.johnsnowlabs.ml.tensorflow.{TensorResources, TensorflowWrapper}
import com.johnsnowlabs.ml.util.{ONNX, Openvino, TensorFlow}
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.annotators.tokenizer.wordpiece.{BasicTokenizer, WordpieceEncoder}
import com.johnsnowlabs.nlp.{ActivationFunction, Annotation}
import org.intel.openvino.Tensor
import org.tensorflow.ndarray.buffer.IntDataBuffer
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.JavaConverters._

/** @param tensorflowWrapper
  *   Bert Model wrapper with TensorFlow Wrapper
  * @param sentenceStartTokenId
  *   Id of sentence start Token
  * @param sentenceEndTokenId
  *   Id of sentence end Token.
  * @param configProtoBytes
  *   Configuration for TensorFlow session
  * @param tags
  *   labels which model was trained with in order
  * @param signatures
  *   TF v2 signatures in Spark NLP
  */
private[johnsnowlabs] class BertClassification(
    val tensorflowWrapper: Option[TensorflowWrapper],
    val onnxWrapper: Option[OnnxWrapper],
    val openvinoWrapper: Option[OpenvinoWrapper],
    val sentenceStartTokenId: Int,
    val sentenceEndTokenId: Int,
    configProtoBytes: Option[Array[Byte]] = None,
    tags: Map[String, Int],
    signatures: Option[Map[String, String]] = None,
    vocabulary: Map[String, Int],
    threshold: Float = 0.5f)
    extends Serializable
    with XXXForClassification {

  protected val logger: Logger = LoggerFactory.getLogger("BertClassification")
  val _tfBertSignatures: Map[String, String] = signatures.getOrElse(ModelSignatureManager.apply())

  protected val sentencePadTokenId = 0
  protected val sigmoidThreshold: Float = threshold
  val detectedEngine: String =
    if (tensorflowWrapper.isDefined) TensorFlow.name
    else if (onnxWrapper.isDefined) ONNX.name
    else if (openvinoWrapper.isDefined) Openvino.name
    else TensorFlow.name
  private val onnxSessionOptions: Map[String, String] = new OnnxSession().getSessionOptions

  private def padArrayWithZeros(arr: Array[Int], maxLength: Int): Array[Int] = {
    if (arr.length >= maxLength) {
      arr
    } else {
      arr ++ Array.fill(maxLength - arr.length)(0)
    }
  }

  def tokenizeWithAlignment(
      sentences: Seq[TokenizedSentence],
      maxSeqLength: Int,
      caseSensitive: Boolean): Seq[WordpieceTokenizedSentence] = {

    val basicTokenizer = new BasicTokenizer(caseSensitive)
    val encoder = new WordpieceEncoder(vocabulary)

    sentences.map { tokenIndex =>
      // filter empty and only whitespace tokens
      val bertTokens =
        tokenIndex.indexedTokens.filter(x => x.token.nonEmpty && !x.token.equals(" ")).map {
          token =>
            val content = if (caseSensitive) token.token else token.token.toLowerCase()
            val sentenceBegin = token.begin
            val sentenceEnd = token.end
            val sentenceIndex = tokenIndex.sentenceIndex
            val result = basicTokenizer.tokenize(
              Sentence(content, sentenceBegin, sentenceEnd, sentenceIndex))
            if (result.nonEmpty) result.head else IndexedToken("")
        }
      val wordpieceTokens = bertTokens.flatMap(token => encoder.encode(token)).take(maxSeqLength)
      WordpieceTokenizedSentence(wordpieceTokens)
    }
  }

  def tokenizeSeqString(
      candidateLabels: Seq[String],
      maxSeqLength: Int,
      caseSensitive: Boolean): Seq[WordpieceTokenizedSentence] = {

    val basicTokenizer = new BasicTokenizer(caseSensitive)
    val encoder = new WordpieceEncoder(vocabulary)

    val labelsToSentences = candidateLabels.map { s => Sentence(s, 0, s.length - 1, 0) }

    labelsToSentences.map(label => {
      val tokens = basicTokenizer.tokenize(label)
      val wordpieceTokens = tokens.flatMap(token => encoder.encode(token)).take(maxSeqLength)
      WordpieceTokenizedSentence(wordpieceTokens)
    })

  }

  def tokenizeDocument(
      docs: Seq[Annotation],
      maxSeqLength: Int,
      caseSensitive: Boolean): Seq[WordpieceTokenizedSentence] = {

    // we need the original form of the token
    // let's lowercase if needed right before the encoding
    val basicTokenizer = new BasicTokenizer(caseSensitive = true, hasBeginEnd = false)
    val encoder = new WordpieceEncoder(vocabulary)
    val sentences = docs.map { s => Sentence(s.result, s.begin, s.end, 0) }

    sentences.map { sentence =>
      val tokens = basicTokenizer.tokenize(sentence)

      val wordpieceTokens = if (caseSensitive) {
        tokens.flatMap(token => encoder.encode(token))
      } else {
        // now we can lowercase the tokens since we have the original form already
        val normalizedTokens =
          tokens.map(x => IndexedToken(x.token.toLowerCase(), x.begin, x.end))
        val normalizedWordPiece = normalizedTokens.flatMap(token => encoder.encode(token))

        normalizedWordPiece.map { t =>
          val orgToken = tokens
            .find(org => t.begin == org.begin && t.isWordStart)
            .map(x => x.token)
            .getOrElse(t.token)
          TokenPiece(t.wordpiece, orgToken, t.pieceId, t.isWordStart, t.begin, t.end)
        }
      }

      WordpieceTokenizedSentence(wordpieceTokens)
    }
  }

  private def getRawScoresWithOv(batch: Seq[Array[Int]], maxSentenceLength: Int): Array[Float] = {

    val batchLength = batch.length
    val shape = Array(batchLength, maxSentenceLength)
    val (tokenTensors, maskTensors) =
      PrepareEmbeddings.prepareOvLongBatchTensors(batch, maxSentenceLength, batchLength)
    val segmentTensors = new Tensor(shape, Array.fill(batchLength * maxSentenceLength)(0L))

    val inferRequest = openvinoWrapper.get.getCompiledModel().create_infer_request()
    inferRequest.set_tensor("input_ids", tokenTensors)
    inferRequest.set_tensor("attention_mask", maskTensors)
    inferRequest.set_tensor("token_type_ids", segmentTensors)

    inferRequest.infer()

    try {
      try {
        inferRequest
          .get_tensor("logits")
          .data()
      }
    } catch {
      case e: Exception =>
        // Log the exception as a warning
        logger.warn("Exception in getRawScoresWithOnnx", e)
        // Rethrow the exception to propagate it further
        throw e
    }

  }

  def tag(batch: Seq[Array[Int]]): Seq[Array[Array[Float]]] = {
    val batchLength = batch.length
    val maxSentenceLength = batch.map(encodedSentence => encodedSentence.length).max

    val rawScores = detectedEngine match {
      case ONNX.name =>
        getRawScoresWithOnnx(batch, maxSentenceLength)
      case TensorFlow.name => getRawScoresWithTF(batch, maxSentenceLength)
      case Openvino.name => getRawScoresWithOv(batch, maxSentenceLength)

    }

    val dim = rawScores.length / (batchLength * maxSentenceLength)
    val batchScores: Array[Array[Array[Float]]] = rawScores
      .grouped(dim)
      .map(scores => calculateSoftmax(scores))
      .toArray
      .grouped(maxSentenceLength)
      .toArray

    batchScores
  }

  private def getRawScoresWithTF(batch: Seq[Array[Int]], maxSentenceLength: Int): Array[Float] = {
    val batchLength = batch.length
    val tensors = new TensorResources()

    val tokenBuffers: IntDataBuffer = tensors.createIntBuffer(batchLength * maxSentenceLength)
    val maskBuffers: IntDataBuffer = tensors.createIntBuffer(batchLength * maxSentenceLength)
    val segmentBuffers: IntDataBuffer = tensors.createIntBuffer(batchLength * maxSentenceLength)

    // [nb of encoded sentences , maxSentenceLength]
    val shape = Array(batch.length.toLong, maxSentenceLength)

    batch.zipWithIndex
      .foreach { case (sentence, idx) =>
        val offset = idx * maxSentenceLength
        tokenBuffers.offset(offset).write(sentence)
        maskBuffers.offset(offset).write(sentence.map(x => if (x == 0) 0 else 1))
        segmentBuffers.offset(offset).write(Array.fill(maxSentenceLength)(0))
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
        _tfBertSignatures.getOrElse(ModelSignatureConstants.InputIds.key, "missing_input_id_key"),
        tokenTensors)
      .feed(
        _tfBertSignatures
          .getOrElse(ModelSignatureConstants.AttentionMask.key, "missing_input_mask_key"),
        maskTensors)
      .feed(
        _tfBertSignatures
          .getOrElse(ModelSignatureConstants.TokenTypeIds.key, "missing_segment_ids_key"),
        segmentTensors)
      .fetch(_tfBertSignatures
        .getOrElse(ModelSignatureConstants.LogitsOutput.key, "missing_logits_key"))

    val outs = runner.run().asScala
    val rawScores = TensorResources.extractFloats(outs.head)

    outs.foreach(_.close())
    tensors.clearSession(outs)
    tensors.clearTensors()

    rawScores
  }

  private def getRawScoresWithOnnx(
      batch: Seq[Array[Int]],
      maxSentenceLength: Int): Array[Float] = {

    // [nb of encoded sentences , maxSentenceLength]
    val (runner, env) = onnxWrapper.get.getSession(onnxSessionOptions)

    val tokenTensors =
      OnnxTensor.createTensor(env, batch.map(x => x.map(x => x.toLong)).toArray)
    val maskTensors =
      OnnxTensor.createTensor(
        env,
        batch.map(sentence => sentence.map(x => if (x == 0L) 0L else 1L)).toArray)

    val segmentTensors =
      OnnxTensor.createTensor(env, batch.map(x => Array.fill(maxSentenceLength)(0L)).toArray)

    val inputs =
      Map(
        "input_ids" -> tokenTensors,
        "attention_mask" -> maskTensors,
        "token_type_ids" -> segmentTensors).asJava

    try {
      val results = runner.run(inputs)
      try {
        val embeddings = results
          .get("logits")
          .get()
          .asInstanceOf[OnnxTensor]
          .getFloatBuffer
          .array()

        embeddings
      } finally if (results != null) results.close()
    } catch {
      case e: Exception =>
        // Handle exceptions by logging or other means.
        e.printStackTrace()
        Array.empty[Float] // Return an empty array or appropriate error handling
    } finally {
      // Close tensors outside the try-catch to avoid repeated null checks.
      // These resources are initialized before the try-catch, so they should be closed here.
      tokenTensors.close()
      maskTensors.close()
      segmentTensors.close()
    }

  }

  def tagSequence(batch: Seq[Array[Int]], activation: String): Array[Array[Float]] = {

    val batchLength = batch.length
    val maxSentenceLength = batch.map(encodedSentence => encodedSentence.length).max
    val rawScores = detectedEngine match {
      case ONNX.name =>
        getRawScoresWithOnnx(batch, maxSentenceLength)
      case Openvino.name => getRawScoresWithOv(batch, maxSentenceLength)
      case TensorFlow.name => getRawScoresWithTF(batch, maxSentenceLength)
    }

    val dim = rawScores.length / batchLength
    val batchScores: Array[Array[Float]] =
      rawScores
        .grouped(dim)
        .map(scores =>
          activation match {
            case ActivationFunction.softmax => calculateSoftmax(scores)
            case ActivationFunction.sigmoid => calculateSigmoid(scores)
            case _ => calculateSoftmax(scores)
          })
        .toArray

    batchScores
  }

  def computeZeroShotLogitsWithOv(
      batch: Seq[Array[Int]],
      maxSentenceLength: Int): Array[Float] = {
    val batchLength = batch.length
    val shape = Array(batchLength, maxSentenceLength)
    val (tokenTensors, maskTensors) =
      PrepareEmbeddings.prepareOvLongBatchTensors(batch, maxSentenceLength, batchLength)

    // Initialize the segment tensor as an array of arrays
    val segmentTensor = batch
      .map(sentence =>
        sentence.indices
          .map(i =>
            if (i < sentence.indexOf(sentenceEndTokenId)) 0L
            else if (i == sentence.indexOf(sentenceEndTokenId)) 1L
            else 1L)
          .toArray)
      .toArray

    val segmentTensors = new Tensor(Array(batch.length, maxSentenceLength), segmentTensor.flatten)

    val inferRequest = openvinoWrapper.get.getCompiledModel().create_infer_request()
    inferRequest.set_tensor("input_ids", tokenTensors)
    inferRequest.set_tensor("attention_mask", maskTensors)
    inferRequest.set_tensor("token_type_ids", segmentTensors)

    inferRequest.infer()

    try {
      try {
        inferRequest
          .get_tensor("logits")
          .data()
      }
    } catch {
      case e: Exception =>
        // Log the exception as a warning
        logger.warn("Exception in getRawScoresWithOnnx", e)
        // Rethrow the exception to propagate it further
        throw e
    }

  }

  def computeZeroShotLogitsWithONNX(
      batch: Seq[Array[Int]],
      maxSentenceLength: Int): Array[Float] = {

    val (runner, env) = onnxWrapper.get.getSession(onnxSessionOptions)

    val tokenTensors =
      OnnxTensor.createTensor(env, batch.map(x => x.map(x => x.toLong)).toArray)
    val maskTensors =
      OnnxTensor.createTensor(
        env,
        batch.map(sentence => sentence.map(x => if (x == 0L) 0L else 1L)).toArray)

    val segmentTensors =
      OnnxTensor.createTensor(
        env,
        batch
          .map(sentence =>
            sentence.indices
              .map(i =>
                if (i < sentence.indexOf(sentenceEndTokenId)) 0L
                else if (i == sentence.indexOf(sentenceEndTokenId)) 1L
                else 1L)
              .toArray)
          .toArray)

    val inputs =
      Map(
        "input_ids" -> tokenTensors,
        "attention_mask" -> maskTensors,
        "token_type_ids" -> segmentTensors).asJava

    try {
      val results = runner.run(inputs)
      try {
        val embeddings = results
          .get("logits")
          .get()
          .asInstanceOf[OnnxTensor]
          .getFloatBuffer
          .array()
        tokenTensors.close()
        maskTensors.close()
        segmentTensors.close()

        embeddings
      } finally if (results != null) results.close()
    }

  }

  def computeZeroShotLogitsWithTF(
      batch: Seq[Array[Int]],
      maxSentenceLength: Int): Array[Float] = {

    val tensors = new TensorResources()
    val batchLength = batch.length

    val tokenBuffers: IntDataBuffer = tensors.createIntBuffer(batchLength * maxSentenceLength)
    val maskBuffers: IntDataBuffer = tensors.createIntBuffer(batchLength * maxSentenceLength)
    val segmentBuffers: IntDataBuffer = tensors.createIntBuffer(batchLength * maxSentenceLength)

    // [nb of encoded sentences , maxSentenceLength]
    val shape = Array(batch.length.toLong, maxSentenceLength)

    batch.zipWithIndex
      .foreach { case (sentence, idx) =>
        val offset = idx * maxSentenceLength
        tokenBuffers.offset(offset).write(sentence)
        maskBuffers.offset(offset).write(sentence.map(x => if (x == 0) 0 else 1))
        val sentenceEndTokenIndex = sentence.indexOf(sentenceEndTokenId)
        segmentBuffers
          .offset(offset)
          .write(
            sentence.indices
              .map(i =>
                if (i < sentenceEndTokenIndex) 0
                else if (i == sentenceEndTokenIndex) 1
                else 1)
              .toArray)
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
        _tfBertSignatures.getOrElse(ModelSignatureConstants.InputIds.key, "missing_input_id_key"),
        tokenTensors)
      .feed(
        _tfBertSignatures
          .getOrElse(ModelSignatureConstants.AttentionMask.key, "missing_input_mask_key"),
        maskTensors)
      .feed(
        _tfBertSignatures
          .getOrElse(ModelSignatureConstants.TokenTypeIds.key, "missing_segment_ids_key"),
        segmentTensors)
      .fetch(_tfBertSignatures
        .getOrElse(ModelSignatureConstants.LogitsOutput.key, "missing_logits_key"))

    val outs = runner.run().asScala
    val rawScores = TensorResources.extractFloats(outs.head)

    outs.foreach(_.close())
    tensors.clearSession(outs)
    tensors.clearTensors()

    rawScores
  }

  def tagZeroShotSequence(
      batch: Seq[Array[Int]],
      entailmentId: Int,
      contradictionId: Int,
      activation: String): Array[Array[Float]] = {

    val maxSentenceLength = batch.map(encodedSentence => encodedSentence.length).max
    val paddedBatch = batch.map(arr => padArrayWithZeros(arr, maxSentenceLength))
    val batchLength = paddedBatch.length

    val rawScores = detectedEngine match {
      case ONNX.name => computeZeroShotLogitsWithONNX(paddedBatch, maxSentenceLength)
      case Openvino.name => computeZeroShotLogitsWithOv(paddedBatch, maxSentenceLength)
      case _ => computeZeroShotLogitsWithTF(paddedBatch, maxSentenceLength)
    }

    val dim = rawScores.length / batchLength
    rawScores
      .grouped(dim)
      .toArray
  }

  def tagSpan(batch: Seq[Array[Int]]): (Array[Array[Float]], Array[Array[Float]]) = {

    val batchLength = batch.length
    val maxSentenceLength = batch.map(encodedSentence => encodedSentence.length).max
    val (startLogits, endLogits) = detectedEngine match {
      case ONNX.name => computeLogitsWithOnnx(batch, maxSentenceLength)
      case Openvino.name => computeLogitsWithOv(batch, maxSentenceLength)
      case _ => computeLogitsWithTF(batch, maxSentenceLength)
    }

    val endDim = endLogits.length / batchLength
    val endScores: Array[Array[Float]] =
      endLogits.grouped(endDim).map(scores => calculateSoftmax(scores)).toArray

    val startDim = startLogits.length / batchLength
    val startScores: Array[Array[Float]] =
      startLogits.grouped(startDim).map(scores => calculateSoftmax(scores)).toArray

    (startScores, endScores)
  }

  private def computeLogitsWithTF(
      batch: Seq[Array[Int]],
      maxSentenceLength: Int): (Array[Float], Array[Float]) = {
    val batchLength = batch.length
    val tensors = new TensorResources()

    val tokenBuffers: IntDataBuffer = tensors.createIntBuffer(batchLength * maxSentenceLength)
    val maskBuffers: IntDataBuffer = tensors.createIntBuffer(batchLength * maxSentenceLength)
    val segmentBuffers: IntDataBuffer = tensors.createIntBuffer(batchLength * maxSentenceLength)

    // [nb of encoded sentences , maxSentenceLength]
    val shape = Array(batch.length.toLong, maxSentenceLength)

    batch.zipWithIndex
      .foreach { case (sentence, idx) =>
        val offset = idx * maxSentenceLength
        tokenBuffers.offset(offset).write(sentence)
        maskBuffers.offset(offset).write(sentence.map(x => if (x == 0) 0 else 1))
        var firstSeq = true
        segmentBuffers
          .offset(offset)
          .write(sentence.map { x =>
            if (firstSeq) {
              if (x == sentenceEndTokenId) {
                firstSeq = false
                1
              } else {
                0
              }
            } else 1
          })
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
        _tfBertSignatures.getOrElse(ModelSignatureConstants.InputIds.key, "missing_input_id_key"),
        tokenTensors)
      .feed(
        _tfBertSignatures.getOrElse(
          ModelSignatureConstants.AttentionMask.key,
          "missing_input_mask_key"),
        maskTensors)
      .feed(
        _tfBertSignatures.getOrElse(
          ModelSignatureConstants.TokenTypeIds.key,
          "missing_segment_ids_key"),
        segmentTensors)
      .fetch(_tfBertSignatures
        .getOrElse(ModelSignatureConstants.EndLogitsOutput.key, "missing_end_logits_key"))
      .fetch(_tfBertSignatures
        .getOrElse(ModelSignatureConstants.StartLogitsOutput.key, "missing_start_logits_key"))

    val outs = runner.run().asScala
    val endLogits = TensorResources.extractFloats(outs.head)
    val startLogits = TensorResources.extractFloats(outs.last)

    outs.foreach(_.close())
    tensors.clearSession(outs)
    tensors.clearTensors()

    (startLogits, endLogits)
  }

  private def computeLogitsWithOv(
      batch: Seq[Array[Int]],
      maxSentenceLength: Int): (Array[Float], Array[Float]) = {
    val batchLength = batch.length
    val shape = Array(batchLength, maxSentenceLength)
    val (tokenTensors, maskTensors) =
      PrepareEmbeddings.prepareOvLongBatchTensors(batch, maxSentenceLength, batchLength)

    // Initialize the segment tensor as an array of arrays
    val segmentTensor = Array.ofDim[Long](batch.length, maxSentenceLength)

    batch.zipWithIndex.foreach { case (sentence, idx) =>
      var firstSeq = true
      val segments: Array[Long] = sentence.map { x =>
        if (firstSeq) {
          if (x == sentenceEndTokenId) {
            firstSeq = false
            1L
          } else {
            0L
          }
        } else {
          1L
        }
      }

      segmentTensor(idx) = segments
    }

    val segmentTensors = new Tensor(Array(batch.length, maxSentenceLength), segmentTensor.flatten)

    val inferRequest = openvinoWrapper.get.getCompiledModel().create_infer_request()
    inferRequest.set_tensor("input_ids", tokenTensors)
    inferRequest.set_tensor("attention_mask", maskTensors)
    inferRequest.set_tensor("token_type_ids", segmentTensors)

    inferRequest.infer()

    try {
      try {
        val startLogits = inferRequest
          .get_tensor("start_logits")
          .data()
        val endLogits = inferRequest
          .get_tensor("end_logits")
          .data()

        (startLogits.slice(1, startLogits.length), endLogits.slice(1, endLogits.length))
      }
    } catch {
      case e: Exception =>
        // Log the exception as a warning
        logger.warn("Exception in getRawScoresWithOnnx", e)
        // Rethrow the exception to propagate it further
        throw e
    }

  }

  private def computeLogitsWithOnnx(
      batch: Seq[Array[Int]],
      maxSentenceLength: Int): (Array[Float], Array[Float]) = {
    // [nb of encoded sentences , maxSentenceLength]
    val (runner, env) = onnxWrapper.get.getSession(onnxSessionOptions)

    val tokenTensors =
      OnnxTensor.createTensor(env, batch.map(x => x.map(x => x.toLong)).toArray)
    val maskTensors =
      OnnxTensor.createTensor(
        env,
        batch.map(sentence => sentence.map(x => if (x == 0L) 0L else 1L)).toArray)

    val segmentTensors =
      OnnxTensor.createTensor(env, batch.map(x => Array.fill(maxSentenceLength)(0L)).toArray)

    val inputs =
      Map(
        "input_ids" -> tokenTensors,
        "attention_mask" -> maskTensors,
        "token_type_ids" -> segmentTensors).asJava

    try {
      val output = runner.run(inputs)
      try {
        val startLogits = output
          .get("start_logits")
          .get()
          .asInstanceOf[OnnxTensor]
          .getFloatBuffer
          .array()

        val endLogits = output
          .get("end_logits")
          .get()
          .asInstanceOf[OnnxTensor]
          .getFloatBuffer
          .array()

        tokenTensors.close()
        maskTensors.close()
        segmentTensors.close()

        (startLogits, endLogits)
      } finally if (output != null) output.close()
    } catch {
      case e: Exception =>
        // Log the exception as a warning
        logger.warn("Exception: ", e)
        // Rethrow the exception to propagate it further
        throw e
    }
  }

  def findIndexedToken(
      tokenizedSentences: Seq[TokenizedSentence],
      sentence: (WordpieceTokenizedSentence, Int),
      tokenPiece: TokenPiece): Option[IndexedToken] = {
    tokenizedSentences(sentence._2).indexedTokens.find(p => p.begin == tokenPiece.begin)
  }

}
