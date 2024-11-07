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

import ai.onnxruntime.{OnnxTensor, OrtEnvironment}
import com.johnsnowlabs.ml.ai.util.PrepareEmbeddings
import com.johnsnowlabs.ml.onnx.{OnnxSession, OnnxWrapper}
import com.johnsnowlabs.ml.openvino.OpenvinoWrapper
import com.johnsnowlabs.ml.tensorflow.sentencepiece.{SentencePieceWrapper, SentencepieceEncoder}
import com.johnsnowlabs.ml.tensorflow.sign.{ModelSignatureConstants, ModelSignatureManager}
import com.johnsnowlabs.ml.tensorflow.{TensorResources, TensorflowWrapper}
import com.johnsnowlabs.ml.util.{ONNX, Openvino, TensorFlow}
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.annotators.tokenizer.wordpiece.BasicTokenizer
import com.johnsnowlabs.nlp.{ActivationFunction, Annotation}
import org.tensorflow.ndarray.buffer.{IntDataBuffer, LongDataBuffer}
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.JavaConverters._

/** @param tensorflowWrapper
  *   CamemBERT Model wrapper with TensorFlow Wrapper
  * @param spp
  *   XlmRoberta SentencePiece model with SentencePieceWrapper
  * @param configProtoBytes
  *   Configuration for TensorFlow session
  * @param tags
  *   labels which model was trained with in order
  * @param signatures
  *   TF v2 signatures in Spark NLP
  */
private[johnsnowlabs] class CamemBertClassification(
    val tensorflowWrapper: Option[TensorflowWrapper],
    val onnxWrapper: Option[OnnxWrapper],
    val openvinoWrapper: Option[OpenvinoWrapper],
    val spp: SentencePieceWrapper,
    configProtoBytes: Option[Array[Byte]] = None,
    tags: Map[String, Int],
    signatures: Option[Map[String, String]] = None,
    threshold: Float = 0.5f)
    extends Serializable
    with XXXForClassification {

  protected val logger: Logger = LoggerFactory.getLogger("CamemBertClassification")
  val _tfCamemBertSignatures: Map[String, String] =
    signatures.getOrElse(ModelSignatureManager.apply())
  val detectedEngine: String =
    if (tensorflowWrapper.isDefined) TensorFlow.name
    else if (onnxWrapper.isDefined) ONNX.name
    else if (openvinoWrapper.isDefined) Openvino.name
    else TensorFlow.name
  private val onnxSessionOptions: Map[String, String] = new OnnxSession().getSessionOptions

  /** HACK: These tokens were added by fairseq but don't seem to be actually used when duplicated
    * in the actual # sentencepiece vocabulary (this is the case for '''<s>''' and '''</s>''')
    * '''<s>NOTUSED": 0''','''"<pad>": 1''', '''"</s>NOTUSED": 2''', '''"<unk>": 3'''
    */
  private val pieceIdOffset: Int = 4
  protected val sentenceStartTokenId: Int = spp.getSppModel.pieceToId("<s>") + pieceIdOffset
  protected val sentenceEndTokenId: Int = spp.getSppModel.pieceToId("</s>") + pieceIdOffset
  protected val sentencePadTokenId: Int = spp.getSppModel.pieceToId("<pad>") + pieceIdOffset
  // unlike other models the delimiter id is correct and does not need pieceIdOffset
  // subtracting pieceIdOffset here to make up for adding it later in SP class
  protected val sentencePieceDelimiterId: Int = spp.getSppModel.pieceToId("▁") - pieceIdOffset
  protected val sigmoidThreshold: Float = threshold

  def tokenizeWithAlignment(
      sentences: Seq[TokenizedSentence],
      maxSeqLength: Int,
      caseSensitive: Boolean): Seq[WordpieceTokenizedSentence] = {

    val encoder =
      new SentencepieceEncoder(
        spp,
        caseSensitive,
        sentencePieceDelimiterId,
        pieceIdOffset = pieceIdOffset)

    val sentenceTokenPieces = sentences.map { s =>
      val trimmedSentence = s.indexedTokens.take(maxSeqLength - 2)
      val wordpieceTokens =
        trimmedSentence.flatMap(token => encoder.encode(token)).take(maxSeqLength)
      WordpieceTokenizedSentence(wordpieceTokens)
    }
    sentenceTokenPieces
  }

  def tokenizeSeqString(
      candidateLabels: Seq[String],
      maxSeqLength: Int,
      caseSensitive: Boolean): Seq[WordpieceTokenizedSentence] = {

    val basicTokenizer = new BasicTokenizer(caseSensitive)
    val encoder =
      new SentencepieceEncoder(spp, caseSensitive, sentencePieceDelimiterId, pieceIdOffset = 1)

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

    val encoder =
      new SentencepieceEncoder(
        spp,
        caseSensitive,
        sentencePieceDelimiterId,
        pieceIdOffset = pieceIdOffset)

    val sentences = docs.map { s => Sentence(s.result, s.begin, s.end, 0) }

    val sentenceTokenPieces = sentences.map { s =>
      val wordpieceTokens = encoder.encodeSentence(s, maxLength = maxSeqLength).take(maxSeqLength)
      WordpieceTokenizedSentence(wordpieceTokens)
    }
    sentenceTokenPieces
  }

  def tag(batch: Seq[Array[Int]]): Seq[Array[Array[Float]]] = {
    val batchLength = batch.length
    val maxSentenceLength = batch.map(encodedSentence => encodedSentence.length).max

    val rawScores = detectedEngine match {
      case ONNX.name => getRawScoresWithOnnx(batch)
      case Openvino.name => getRawScoresWithOv(batch, maxSentenceLength)
      case _ => getRawScoresWithTF(batch, maxSentenceLength)
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
    val tensors = new TensorResources()
    val maxSentenceLength = batch.map(encodedSentence => encodedSentence.length).max
    val batchLength = batch.length

    val tokenBuffers: LongDataBuffer = tensors.createLongBuffer(batchLength * maxSentenceLength)
    val maskBuffers: LongDataBuffer = tensors.createLongBuffer(batchLength * maxSentenceLength)

    // [nb of encoded sentences , maxSentenceLength]
    val shape = Array(batch.length.toLong, maxSentenceLength)

    batch.zipWithIndex
      .foreach { case (sentence, idx) =>
        val sentenceLong = sentence.map(x => x.toLong)
        val offset = idx * maxSentenceLength
        tokenBuffers.offset(offset).write(sentenceLong)
        maskBuffers
          .offset(offset)
          .write(sentence.map(x => if (x == sentencePadTokenId) 0L else 1L))
      }

    val runner = tensorflowWrapper.get
      .getTFSessionWithSignature(configProtoBytes = configProtoBytes, initAllTables = false)
      .runner

    val tokenTensors = tensors.createLongBufferTensor(shape, tokenBuffers)
    val maskTensors = tensors.createLongBufferTensor(shape, maskBuffers)

    runner
      .feed(
        _tfCamemBertSignatures
          .getOrElse(ModelSignatureConstants.InputIds.key, "missing_input_id_key"),
        tokenTensors)
      .feed(
        _tfCamemBertSignatures
          .getOrElse(ModelSignatureConstants.AttentionMask.key, "missing_input_mask_key"),
        maskTensors)
      .fetch(_tfCamemBertSignatures
        .getOrElse(ModelSignatureConstants.LogitsOutput.key, "missing_logits_key"))

    val outs = runner.run().asScala
    val rawScores = TensorResources.extractFloats(outs.head)

    outs.foreach(_.close())
    tensors.clearSession(outs)
    tensors.clearTensors()

    rawScores
  }

  private def getRawScoresWithOnnx(batch: Seq[Array[Int]]): Array[Float] = {
    val (runner, env) = onnxWrapper.get.getSession(onnxSessionOptions)
    val (tokenTensors, maskTensors) = initializeOnnxTensorResources(batch, env)
    val inputs =
      Map("input_ids" -> tokenTensors, "attention_mask" -> maskTensors).asJava

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
    }
  }

  private def getRawScoresWithOv(batch: Seq[Array[Int]], maxSentenceLength: Int): Array[Float] = {

    val batchLength = batch.length
    val shape = Array(batchLength, maxSentenceLength)
    val (tokenTensors, maskTensors) =
      PrepareEmbeddings.prepareOvLongBatchTensors(batch, maxSentenceLength, batchLength)

    val inferRequest = openvinoWrapper.get.getCompiledModel().create_infer_request()
    inferRequest.set_tensor("input_ids", tokenTensors)
    inferRequest.set_tensor("attention_mask", maskTensors)
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

  def tagSequence(batch: Seq[Array[Int]], activation: String): Array[Array[Float]] = {

    val maxSentenceLength = batch.map(encodedSentence => encodedSentence.length).max
    val batchLength = batch.length

    val rawScores = detectedEngine match {
      case ONNX.name => getRawScoresWithOnnx(batch)
      case Openvino.name => getRawScoresWithOv(batch, maxSentenceLength)
      case _ => getRawScoresWithTF(batch, maxSentenceLength)
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

  def tagZeroShotSequence(
      batch: Seq[Array[Int]],
      entailmentId: Int,
      contradictionId: Int,
      activation: String): Array[Array[Float]] = {

    val maxSentenceLength = batch.map(encodedSentence => encodedSentence.length).max
    val paddedBatch = batch.map(arr => padArrayWithZeros(arr, maxSentenceLength))
    val batchLength = paddedBatch.length

    val rawScores = detectedEngine match {
      case TensorFlow.name => computeZeroShotLogitsWithTF(paddedBatch, maxSentenceLength)
      case ONNX.name => computeZeroShotLogitsWithONNX(paddedBatch)
    }

    val dim = rawScores.length / batchLength
    rawScores
      .grouped(dim)
      .toArray
  }

  def computeZeroShotLogitsWithONNX(batch: Seq[Array[Int]]): Array[Float] = {
    val (runner, env) = onnxWrapper.get.getSession(onnxSessionOptions)
    val (tokenTensors, maskTensors) = initializeOnnxTensorResources(batch, env)
    val inputs =
      Map("input_ids" -> tokenTensors, "attention_mask" -> maskTensors).asJava

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

        embeddings
      } finally if (results != null) results.close()
    }
  }

  def computeZeroShotLogitsWithTF(
      batch: Seq[Array[Int]],
      maxSentenceLength: Int): Array[Float] = {
    val tensors = new TensorResources()
    val (tokenBuffers, maskBuffers, segmentBuffers) =
      initializeTFIntTensorResources(batch, tensors, maxSentenceLength)
    // [nb of encoded sentences , maxSentenceLength]
    val shape = Array(batch.length.toLong, maxSentenceLength)

    batch.zipWithIndex
      .foreach { case (sentence, idx) =>
        val offset = idx * maxSentenceLength
        tokenBuffers.offset(offset).write(sentence)
        maskBuffers
          .offset(offset)
          .write(sentence.map(x => if (x == sentencePadTokenId) 0 else 1))
        segmentBuffers.offset(offset).write(Array.fill(maxSentenceLength)(0))
      }

    val runner = tensorflowWrapper.get
      .getTFSessionWithSignature(configProtoBytes = configProtoBytes, initAllTables = false)
      .runner

    val tokenTensors = tensors.createIntBufferTensor(shape, tokenBuffers)
    val maskTensors = tensors.createIntBufferTensor(shape, maskBuffers)
    val segmentTensors = tensors.createIntBufferTensor(shape, segmentBuffers)

    runner
      .feed(
        _tfCamemBertSignatures.getOrElse(
          ModelSignatureConstants.InputIds.key,
          "missing_input_id_key"),
        tokenTensors)
      .feed(
        _tfCamemBertSignatures
          .getOrElse(ModelSignatureConstants.AttentionMask.key, "missing_input_mask_key"),
        maskTensors)
      .feed(
        _tfCamemBertSignatures
          .getOrElse(ModelSignatureConstants.TokenTypeIds.key, "missing_segment_ids_key"),
        segmentTensors)
      .fetch(_tfCamemBertSignatures
        .getOrElse(ModelSignatureConstants.LogitsOutput.key, "missing_logits_key"))

    val outs = runner.run().asScala
    val rawScores = TensorResources.extractFloats(outs.head)

    outs.foreach(_.close())
    tensors.clearSession(outs)
    tensors.clearTensors()

    rawScores
  }

  private def padArrayWithZeros(arr: Array[Int], maxLength: Int): Array[Int] = {
    if (arr.length >= maxLength) {
      arr
    } else {
      arr ++ Array.fill(maxLength - arr.length)(0)
    }
  }

  def computeLogitsWithOv(batch: Seq[Array[Int]]): (Array[Float], Array[Float]) = {
    val maxSentenceLength = batch.map(encodedSentence => encodedSentence.length).max
    val batchLength = batch.length
    val shape = Array(batchLength, maxSentenceLength)
    val (tokenTensors, maskTensors) =
      PrepareEmbeddings.prepareOvLongBatchTensors(batch, maxSentenceLength, batchLength)

    val inferRequest = openvinoWrapper.get.getCompiledModel().create_infer_request()
    inferRequest.set_tensor("input_ids", tokenTensors)
    inferRequest.set_tensor("attention_mask", maskTensors)

    inferRequest.infer()

    try {
      try {
        val startLogits = inferRequest
          .get_tensor("start_logits")
          .data()
        val endLogits = inferRequest
          .get_tensor("end_logits")
          .data()

        (startLogits, endLogits)
      }
    } catch {
      case e: Exception =>
        // Log the exception as a warning
        logger.warn("Exception in getRawScoresWithOnnx", e)
        // Rethrow the exception to propagate it further
        throw e
    }

  }

  def tagSpan(batch: Seq[Array[Int]]): (Array[Array[Float]], Array[Array[Float]]) = {
    val batchLength = batch.length
    val maxSentenceLength = batch.map(encodedSentence => encodedSentence.length).max
    val (startLogits, endLogits) = detectedEngine match {
      case ONNX.name => computeLogitsWithOnnx(batch)
      case Openvino.name => computeLogitsWithOv(batch)
      case TensorFlow.name => computeLogitsWithTF(batch, maxSentenceLength)
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

    val tensors = new TensorResources()
    val (tokenBuffers, maskBuffers) =
      initializeTFLongTensorResources(batch, tensors, maxSentenceLength)

    // [nb of encoded sentences , maxSentenceLength]
    val shape = Array(batch.length.toLong, maxSentenceLength)

    // [nb of encoded sentences , maxSentenceLength]
    batch.zipWithIndex
      .foreach { case (sentence, idx) =>
        val sentenceLong = sentence.map(x => x.toLong)
        val offset = idx * maxSentenceLength
        tokenBuffers.offset(offset).write(sentenceLong)
        maskBuffers
          .offset(offset)
          .write(sentence.map(x => if (x == sentencePadTokenId) 0L else 1L))
      }

    val runner = tensorflowWrapper.get
      .getTFSessionWithSignature(configProtoBytes = configProtoBytes, initAllTables = false)
      .runner

    val tokenTensors = tensors.createLongBufferTensor(shape, tokenBuffers)
    val maskTensors = tensors.createLongBufferTensor(shape, maskBuffers)

    runner
      .feed(
        _tfCamemBertSignatures
          .getOrElse(ModelSignatureConstants.InputIds.key, "missing_input_id_key"),
        tokenTensors)
      .feed(
        _tfCamemBertSignatures
          .getOrElse(ModelSignatureConstants.AttentionMask.key, "missing_input_mask_key"),
        maskTensors)
      .fetch(_tfCamemBertSignatures
        .getOrElse(ModelSignatureConstants.EndLogitsOutput.key, "missing_end_logits_key"))
      .fetch(_tfCamemBertSignatures
        .getOrElse(ModelSignatureConstants.StartLogitsOutput.key, "missing_start_logits_key"))

    val outs = runner.run().asScala
    val endLogits = TensorResources.extractFloats(outs.head)
    val startLogits = TensorResources.extractFloats(outs.last)

    outs.foreach(_.close())
    tensors.clearSession(outs)
    tensors.clearTensors()

    (startLogits, endLogits)
  }

  private def initializeTFLongTensorResources(
      batch: Seq[Array[Int]],
      tensors: TensorResources,
      maxSentenceLength: Int): (LongDataBuffer, LongDataBuffer) = {

    val batchLength = batch.length
    val dim = batchLength * maxSentenceLength
    val tokenBuffers: LongDataBuffer = tensors.createLongBuffer(dim)
    val maskBuffers: LongDataBuffer = tensors.createLongBuffer(dim)
    (tokenBuffers, maskBuffers)
  }

  private def initializeTFIntTensorResources(
      batch: Seq[Array[Int]],
      tensors: TensorResources,
      maxSentenceLength: Int): (IntDataBuffer, IntDataBuffer, IntDataBuffer) = {

    val batchLength = batch.length
    val dim = batchLength * maxSentenceLength
    val tokenBuffers: IntDataBuffer = tensors.createIntBuffer(dim)
    val maskBuffers: IntDataBuffer = tensors.createIntBuffer(dim)
    val segmentBuffers: IntDataBuffer = tensors.createIntBuffer(dim)
    (tokenBuffers, maskBuffers, segmentBuffers)
  }

  private def computeLogitsWithOnnx(batch: Seq[Array[Int]]): (Array[Float], Array[Float]) = {
    val (runner, env) = onnxWrapper.get.getSession(onnxSessionOptions)
    val (tokenTensors, maskTensors) = initializeOnnxTensorResources(batch, env)
    val inputs =
      Map("input_ids" -> tokenTensors, "attention_mask" -> maskTensors).asJava

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

  private def initializeOnnxTensorResources(batch: Seq[Array[Int]], env: OrtEnvironment) = {
    val tokenTensors =
      OnnxTensor.createTensor(env, batch.map(x => x.map(x => x.toLong)).toArray)
    val maskTensors =
      OnnxTensor.createTensor(
        env,
        batch.map(sentence => sentence.map(x => if (x == 0L) 0L else 1L)).toArray)

    (tokenTensors, maskTensors)
  }

  def findIndexedToken(
      tokenizedSentences: Seq[TokenizedSentence],
      sentence: (WordpieceTokenizedSentence, Int),
      tokenPiece: TokenPiece): Option[IndexedToken] = {
    tokenizedSentences(sentence._2).indexedTokens.find(p =>
      p.begin == tokenPiece.begin && tokenPiece.isWordStart)
  }
}
