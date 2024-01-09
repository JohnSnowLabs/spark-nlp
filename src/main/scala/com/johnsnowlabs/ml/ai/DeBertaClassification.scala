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
import com.johnsnowlabs.ml.onnx.{OnnxSession, OnnxWrapper}
import com.johnsnowlabs.ml.tensorflow.sentencepiece.{SentencePieceWrapper, SentencepieceEncoder}
import com.johnsnowlabs.ml.tensorflow.sign.{ModelSignatureConstants, ModelSignatureManager}
import com.johnsnowlabs.ml.tensorflow.{TensorResources, TensorflowWrapper}
import com.johnsnowlabs.ml.util.{ONNX, TensorFlow}
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.{ActivationFunction, Annotation}
import org.tensorflow.ndarray.buffer
import org.tensorflow.ndarray.buffer.{IntDataBuffer, LongDataBuffer}

import scala.collection.JavaConverters._

/** @param tensorflowWrapper
  *   DeBERTa Model v2 & v3 wrapper with TensorFlow Wrapper
  * @param spp
  *   DeBERTa SentencePiece model with SentencePieceWrapper
  * @param configProtoBytes
  *   Configuration for TensorFlow session
  * @param tags
  *   labels which model was trained with in order
  * @param signatures
  *   TF v2 signatures in Spark NLP
  */
private[johnsnowlabs] class DeBertaClassification(
    val tensorflowWrapper: Option[TensorflowWrapper],
    val onnxWrapper: Option[OnnxWrapper],
    val spp: SentencePieceWrapper,
    configProtoBytes: Option[Array[Byte]] = None,
    tags: Map[String, Int],
    signatures: Option[Map[String, String]] = None,
    threshold: Float = 0.5f)
    extends Serializable
    with XXXForClassification {

  val _tfDeBertaSignatures: Map[String, String] =
    signatures.getOrElse(ModelSignatureManager.apply())
  val detectedEngine: String =
    if (tensorflowWrapper.isDefined) TensorFlow.name
    else if (onnxWrapper.isDefined) ONNX.name
    else TensorFlow.name
  private val onnxSessionOptions: Map[String, String] = new OnnxSession().getSessionOptions

  // keys representing the input and output tensors of the DeBERTa model
  protected val sentencePadTokenId: Int = spp.getSppModel.pieceToId("[PAD]")
  protected val sentenceStartTokenId: Int = spp.getSppModel.pieceToId("[CLS]")
  protected val sentenceEndTokenId: Int = spp.getSppModel.pieceToId("[SEP]")

  private val sentencePieceDelimiterId: Int = spp.getSppModel.pieceToId("▁")
  protected val sigmoidThreshold: Float = threshold

  def tokenizeWithAlignment(
      sentences: Seq[TokenizedSentence],
      maxSeqLength: Int,
      caseSensitive: Boolean): Seq[WordpieceTokenizedSentence] = {

    val encoder = new SentencepieceEncoder(spp, caseSensitive, sentencePieceDelimiterId)

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
      caseSensitive: Boolean): Seq[WordpieceTokenizedSentence] = ???

  def tokenizeDocument(
      docs: Seq[Annotation],
      maxSeqLength: Int,
      caseSensitive: Boolean): Seq[WordpieceTokenizedSentence] = {
    val encoder =
      new SentencepieceEncoder(spp, caseSensitive, sentencePieceDelimiterId, pieceIdOffset = 0)

    val sentences = docs.map { s => Sentence(s.result, s.begin, s.end, 0) }

    val sentenceTokenPieces = sentences.map { s =>
      val wordpieceTokens = encoder.encodeSentence(s, maxLength = maxSeqLength).take(maxSeqLength)
      WordpieceTokenizedSentence(wordpieceTokens)
    }
    sentenceTokenPieces
  }

  def tag(batch: Seq[Array[Int]]): Seq[Array[Array[Float]]] = {
    val maxSentenceLength = batch.map(encodedSentence => encodedSentence.length).max
    val batchLength = batch.length

    val rawScores = detectedEngine match {
      case ONNX.name => getRawScoresWithOnnx(batch)
      case _ => getRawScoresWithTF(batch)
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

  private def getRawScoresWithTF(batch: Seq[Array[Int]]): Array[Float] = {
    val tensors = new TensorResources()

    val maxSentenceLength = batch.map(encodedSentence => encodedSentence.length).max
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
        _tfDeBertaSignatures.getOrElse(
          ModelSignatureConstants.InputIds.key,
          "missing_input_id_key"),
        tokenTensors)
      .feed(
        _tfDeBertaSignatures
          .getOrElse(ModelSignatureConstants.AttentionMask.key, "missing_input_mask_key"),
        maskTensors)
      .feed(
        _tfDeBertaSignatures
          .getOrElse(ModelSignatureConstants.TokenTypeIds.key, "missing_segment_ids_key"),
        segmentTensors)
      .fetch(_tfDeBertaSignatures
        .getOrElse(ModelSignatureConstants.LogitsOutput.key, "missing_logits_key"))

    val outs = runner.run().asScala
    val rawScores = TensorResources.extractFloats(outs.head)

    outs.foreach(_.close())
    tensors.clearSession(outs)
    tensors.clearTensors()

    rawScores
  }

  private def getRawScoresWithOnnx(batch: Seq[Array[Int]]): Array[Float] = {

    // [nb of encoded sentences , maxSentenceLength]
    val (runner, env) = onnxWrapper.get.getSession(onnxSessionOptions)

    val tokenTensors =
      OnnxTensor.createTensor(env, batch.map(x => x.map(x => x.toLong)).toArray)
    val maskTensors =
      OnnxTensor.createTensor(
        env,
        batch.map(sentence => sentence.map(x => if (x == 0L) 0L else 1L)).toArray)

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

  def tagSequence(batch: Seq[Array[Int]], activation: String): Array[Array[Float]] = {

    val batchLength = batch.length

    val rawScores = detectedEngine match {
      case ONNX.name => getRawScoresWithOnnx(batch)
      case _ => getRawScoresWithTF(batch)
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
      activation: String): Array[Array[Float]] = ???

  def tagSpan(batch: Seq[Array[Int]]): (Array[Array[Float]], Array[Array[Float]]) = {
    val batchLength = batch.length
    val (startLogits, endLogits) = detectedEngine match {
      case ONNX.name => computeLogitsWithOnnx(batch)
      case _ => computeLogitsWithTF(batch)
    }

    val endDim = endLogits.length / batchLength
    val endScores: Array[Array[Float]] =
      endLogits.grouped(endDim).map(scores => calculateSoftmax(scores)).toArray

    val startDim = startLogits.length / batchLength
    val startScores: Array[Array[Float]] =
      startLogits.grouped(startDim).map(scores => calculateSoftmax(scores)).toArray

    (startScores, endScores)
  }

  private def computeLogitsWithTF(batch: Seq[Array[Int]]): (Array[Float], Array[Float]) = {
    val tensors = new TensorResources()

    val maxSentenceLength = batch.map(encodedSentence => encodedSentence.length).max
    val batchLength = batch.length

    val tokenBuffers: IntDataBuffer = tensors.createIntBuffer(batchLength * maxSentenceLength)
    val maskBuffers: IntDataBuffer = tensors.createIntBuffer(batchLength * maxSentenceLength)

    // [nb of encoded sentences , maxSentenceLength]
    val shape = Array(batch.length.toLong, maxSentenceLength)

    batch.zipWithIndex
      .foreach { case (sentence, idx) =>
        val offset = idx * maxSentenceLength
        tokenBuffers.offset(offset).write(sentence)
        maskBuffers
          .offset(offset)
          .write(sentence.map(x => if (x == sentencePadTokenId) 0 else 1))
      }

    val runner = tensorflowWrapper.get
      .getTFSessionWithSignature(configProtoBytes = configProtoBytes, initAllTables = false)
      .runner

    val tokenTensors = tensors.createIntBufferTensor(shape, tokenBuffers)
    val maskTensors = tensors.createIntBufferTensor(shape, maskBuffers)

    runner
      .feed(
        _tfDeBertaSignatures
          .getOrElse(ModelSignatureConstants.InputIds.key, "missing_input_id_key"),
        tokenTensors)
      .feed(
        _tfDeBertaSignatures
          .getOrElse(ModelSignatureConstants.AttentionMask.key, "missing_input_mask_key"),
        maskTensors)
      .fetch(_tfDeBertaSignatures
        .getOrElse(ModelSignatureConstants.EndLogitsOutput.key, "missing_end_logits_key"))
      .fetch(_tfDeBertaSignatures
        .getOrElse(ModelSignatureConstants.StartLogitsOutput.key, "missing_start_logits_key"))

    val outs = runner.run().asScala
    val endLogits = TensorResources.extractFloats(outs.head)
    val startLogits = TensorResources.extractFloats(outs.last)

    outs.foreach(_.close())
    tensors.clearSession(outs)
    tensors.clearTensors()

    (startLogits, endLogits)
  }

  private def computeLogitsWithOnnx(batch: Seq[Array[Int]]): (Array[Float], Array[Float]) = {
    // [nb of encoded sentences]
    val (runner, env) = onnxWrapper.get.getSession(onnxSessionOptions)

    val tokenTensors =
      OnnxTensor.createTensor(env, batch.map(x => x.map(x => x.toLong)).toArray)
    val maskTensors =
      OnnxTensor.createTensor(
        env,
        batch.map(sentence => sentence.map(x => if (x == 0L) 0L else 1L)).toArray)

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
    }
  }

  def findIndexedToken(
      tokenizedSentences: Seq[TokenizedSentence],
      sentence: (WordpieceTokenizedSentence, Int),
      tokenPiece: TokenPiece): Option[IndexedToken] = {

    tokenizedSentences(sentence._2).indexedTokens.find(p =>
      p.begin == tokenPiece.begin && tokenPiece.isWordStart)
  }

}
