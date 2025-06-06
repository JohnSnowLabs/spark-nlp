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

import ai.onnxruntime.{OnnxTensor, OrtEnvironment, OrtSession}
import com.johnsnowlabs.ml.ai.util.Generation.Generate
import com.johnsnowlabs.ml.onnx.{OnnxSession, OnnxWrapper}
import com.johnsnowlabs.ml.onnx.OnnxWrapper.EncoderDecoderWithoutPastWrappers
import com.johnsnowlabs.ml.onnx.TensorResources.implicits.OnnxSessionResult
import com.johnsnowlabs.ml.openvino.OpenvinoWrapper.{
  EncoderDecoderWithoutPastWrappers => OpenvinoEncoderDecoderWithoutPastWrappers
}
import com.johnsnowlabs.ml.tensorflow.sign.{ModelSignatureConstants, ModelSignatureManager}
import com.johnsnowlabs.ml.tensorflow.{TensorResources, TensorflowWrapper}
import com.johnsnowlabs.ml.util.{ONNX, Openvino, TensorFlow}
import com.johnsnowlabs.nlp.annotators.common.SentenceSplit
import com.johnsnowlabs.nlp.annotators.tokenizer.bpe.{BartTokenizer, BpeTokenizer}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import org.intel.openvino.InferRequest
import org.tensorflow.{Session, Tensor}

import scala.collection.JavaConverters._

/** This class is used to run Bart model for For Sequence Batches of WordpieceTokenizedSentence.
  * Input for this model must be tokenized with a SentencePieceModel,
  *
  * @param tensorflow
  *   BART Model wrapper with TensorFlowWrapper
  * @param configProtoBytes
  *   Configuration for TensorFlow session
  */

private[johnsnowlabs] class Bart(
    val tensorflowWrapper: Option[TensorflowWrapper],
    val onnxWrapper: Option[EncoderDecoderWithoutPastWrappers],
    val openvinoWrapper: Option[OpenvinoEncoderDecoderWithoutPastWrappers],
    configProtoBytes: Option[Array[Byte]] = None,
    signatures: Option[Map[String, String]] = None,
    merges: Map[(String, String), Int],
    vocabulary: Map[String, Int],
    useCache: Boolean = false)
    extends Serializable
    with Generate {

  val bpeTokenizer: BartTokenizer = BpeTokenizer
    .forModel("bart", merges = merges, vocab = vocabulary, padWithSequenceTokens = false)
    .asInstanceOf[BartTokenizer]
  private val _tfBartSignatures: Map[String, String] =
    signatures.getOrElse(ModelSignatureManager.apply())
  private val onnxSessionOptions: Map[String, String] = new OnnxSession().getSessionOptions
  private val paddingTokenId = 1
  private val eosTokenId = 2
  private val vocabSize = 50264
  private var decoderEncoderStateTensorsOV: Option[org.intel.openvino.Tensor] = None
  private var encoderAttentionMaskOV: Option[org.intel.openvino.Tensor] = None

  var tensorDecoder = new TensorResources()
  private var nextStateTensor1: Option[org.tensorflow.Tensor] = None
  private var nextStateTensor2: Option[org.tensorflow.Tensor] = None
  val detectedEngine: String =
    if (tensorflowWrapper.isDefined) TensorFlow.name
    else if (onnxWrapper.isDefined) ONNX.name
    else if (openvinoWrapper.isDefined) Openvino.name
    else TensorFlow.name

  private object OnnxSignatures {
    val encoderInputIDs: String = "input_ids"
    val encoderAttentionMask: String = "attention_mask"

    val encoderOutput: String = "last_hidden_state"

    val decoderInputIDs: String = "input_ids"
    val decoderEncoderAttentionMask: String = "encoder_attention_mask"
    val decoderEncoderState: String = "encoder_hidden_states"

    val decoderOutput: String = "logits"
  }

  private object OpenVinoSignatures {
    val encoderInputIDs: String = "input_ids"
    val encoderAttentionMask: String = "attention_mask"

    val encoderOutput: String = "last_hidden_state"

    val decoderInputIDs: String = "input_ids"
    val decoderEncoderAttentionMask: String = "encoder_attention_mask"
    val decoderEncoderState: String = "encoder_hidden_states"

    val decoderOutput: String = "logits"
  }

  /** @param sentences
    *   Sequence of WordpieceTokenizedSentence
    * @param batchSize
    *   Batch size
    * @param minOutputLength
    *   Minimum length of output
    * @param maxOutputLength
    *   Maximum length of output
    * @param doSample
    *   Whether to sample or not
    * @param temperature
    *   Temperature for sampling
    * @param topK
    *   Top K for sampling
    * @param topP
    *   Top P for sampling
    * @param repetitionPenalty
    *   Repetition penalty for sampling
    * @param noRepeatNgramSize
    *   No repeat ngram size for sampling
    * @param task
    *   Task
    * @param randomSeed
    *   Random seed
    * @param ignoreTokenIds
    *   Ignore token ids
    * @param beamSize
    *   Beam size
    * @return
    */
  def predict(
      sentences: Seq[Annotation],
      batchSize: Int,
      minOutputLength: Int,
      maxOutputLength: Int,
      doSample: Boolean,
      temperature: Double,
      topK: Int,
      topP: Double,
      repetitionPenalty: Double,
      noRepeatNgramSize: Int,
      task: String,
      randomSeed: Option[Long] = None,
      ignoreTokenIds: Array[Int] = Array(),
      beamSize: Int,
      maxInputLength: Int): Seq[Annotation] = {

    val batchDecoder = sentences.grouped(batchSize).toArray.flatMap { batch =>
      val batchSP = encode(batch, task)
      val spIds = tag(
        batchSP,
        minOutputLength,
        maxOutputLength,
        doSample,
        temperature,
        topK,
        topP,
        repetitionPenalty,
        noRepeatNgramSize,
        randomSeed,
        ignoreTokenIds,
        beamSize,
        maxInputLength)

      decode(spIds)

    }

    var sentBegin, nextSentEnd = 0
    val annotations = batchDecoder.zip(sentences).map { case (content, sent) =>
      nextSentEnd += content.length - 1
      val annots = new Annotation(
        annotatorType = AnnotatorType.DOCUMENT,
        begin = sentBegin,
        end = nextSentEnd,
        result = content,
        metadata = sent.metadata)
      sentBegin += nextSentEnd + 1
      annots
    }
    tensorDecoder = new TensorResources()
    nextStateTensor1 = None
    nextStateTensor2 = None
    annotations
  }

  /** @param batch
    *   Sequence of WordpieceTokenizedSentence
    * @param minOutputLength
    *   Minimum length of output
    * @param maxOutputLength
    *   Maximum length of output
    * @param doSample
    *   Whether to sample or not
    * @param temperature
    *   Temperature for sampling
    * @param topK
    *   Top K for sampling
    * @param topP
    *   Top P for sampling
    * @param repetitionPenalty
    *   Repetition penalty for sampling
    * @param noRepeatNgramSize
    *   No repeat ngram size for sampling
    * @param randomSeed
    *   Random seed
    * @param ignoreTokenIds
    *   Ignore token ids
    * @param beamSize
    *   Beam size
    * @return
    *   Sequence of WordpieceTokenizedSentence
    */
  def tag(
      batch: Seq[Array[Int]],
      minOutputLength: Int,
      maxOutputLength: Int,
      doSample: Boolean,
      temperature: Double,
      topK: Int,
      topP: Double,
      repetitionPenalty: Double,
      noRepeatNgramSize: Int,
      randomSeed: Option[Long],
      ignoreTokenIds: Array[Int] = Array(),
      beamSize: Int,
      maxInputLength: Int): Array[Array[Int]] = {

    val ignoreTokenIdsInt = ignoreTokenIds
    val expandedEncoderInputIdsVals =
      batch.flatMap(x => List.fill(beamSize)(x.take(maxInputLength)))
    val sequencesLength = expandedEncoderInputIdsVals.map(x => x.length).toArray
    val maxSentenceLength = sequencesLength.max // - curLen

    val numReturn_sequences = 1
    // from config

    var effectiveBatch_size = 1
    var effectiveBatch_mult = 1

    // set effective batch size and effective batch multiplier according to do_sample
    if (doSample) {
      effectiveBatch_size = expandedEncoderInputIdsVals.length * numReturn_sequences
      effectiveBatch_mult = numReturn_sequences
    } else {
      effectiveBatch_size = expandedEncoderInputIdsVals.length
      effectiveBatch_mult = 1
    }

    val decoderInputs = batch.map(_ => Array(this.eosTokenId)).toArray

    if (detectedEngine == TensorFlow.name) {

      // Run encoder
      val tensorEncoder = new TensorResources()
      val inputDim = expandedEncoderInputIdsVals.length * maxSentenceLength

      val encoderInputBuffers = tensorEncoder.createIntBuffer(inputDim)
      val encoderAttentionMaskBuffers = tensorEncoder.createIntBuffer(inputDim)

      val shape = Array(expandedEncoderInputIdsVals.length.toLong, maxSentenceLength)

      expandedEncoderInputIdsVals.zipWithIndex.foreach { case (tokenIds, idx) =>
        val offset = idx * maxSentenceLength
        val diff = maxSentenceLength - tokenIds.length

        val s = tokenIds.take(maxSentenceLength) ++ Array.fill[Int](diff)(this.paddingTokenId)
        encoderInputBuffers.offset(offset).write(s)
        val mask = s.map(x => if (x != this.paddingTokenId) 1 else 0)
        encoderAttentionMaskBuffers.offset(offset).write(mask)
      }

      val session = tensorflowWrapper.get.getTFSessionWithSignature(
        configProtoBytes = configProtoBytes,
        initAllTables = false,
        savedSignatures = signatures)

      val encoderInputTensors = tensorEncoder.createIntBufferTensor(shape, encoderInputBuffers)
      val encoderAttentionMaskTensors =
        tensorEncoder.createIntBufferTensor(shape, encoderAttentionMaskBuffers)

      val runner = session.runner

      runner
        .feed(
          _tfBartSignatures.getOrElse(
            ModelSignatureConstants.EncoderInputIds.key,
            "missing_encoder_input_ids"),
          encoderInputTensors)
        .feed(
          _tfBartSignatures.getOrElse(
            ModelSignatureConstants.EncoderAttentionMask.key,
            "missing_encoder_attention_mask"),
          encoderAttentionMaskTensors)
        .fetch(
          _tfBartSignatures
            .getOrElse(
              ModelSignatureConstants.CachedEncoderOutput.key,
              "missing_last_hidden_state"))

      val encoderOuts = runner.run().asScala
      val encoderOutsFloats = TensorResources.extractFloats(encoderOuts.head)
      val dim = encoderOutsFloats.length / inputDim
      val encoderOutsBatch =
        encoderOutsFloats.grouped(dim).toArray.grouped(maxSentenceLength).toArray

      encoderOuts.foreach(_.close())

      // Run decoder
      val decoderEncoderStateTensorResources = new TensorResources()
      val decoderEncoderStateBuffers =
        decoderEncoderStateTensorResources.createFloatBuffer(
          expandedEncoderInputIdsVals.length * maxSentenceLength * dim)
      expandedEncoderInputIdsVals.zipWithIndex.foreach { case (_, index) =>
        var offset = index * maxSentenceLength * dim
        encoderOutsBatch(index).foreach(encoderOutput => {
          decoderEncoderStateBuffers.offset(offset).write(encoderOutput)
          offset += dim
        })
      }

      val decoderEncoderStateTensors = tensorEncoder.createFloatBufferTensor(
        Array(expandedEncoderInputIdsVals.length, maxSentenceLength, dim),
        decoderEncoderStateBuffers)

      val modelOutputs = generate(
        batch,
        Left(decoderEncoderStateTensors),
        Left(encoderAttentionMaskTensors),
        decoderInputs,
        maxOutputLength,
        minOutputLength,
        doSample,
        beamSize,
        1,
        temperature,
        topK,
        topP,
        repetitionPenalty,
        noRepeatNgramSize,
        this.vocabSize,
        this.eosTokenId,
        this.paddingTokenId,
        randomSeed,
        ignoreTokenIdsInt,
        Left(session))

      tensorEncoder.clearTensors()
      tensorEncoder.clearSession(encoderOuts)
      decoderEncoderStateTensorResources.clearTensors()
      decoderEncoderStateTensors.close()
      encoderAttentionMaskTensors.close()
      encoderInputTensors.close()
      if (useCache) {
        tensorDecoder.clearTensors()
        nextStateTensor1 = None
        nextStateTensor2 = None
      }
      modelOutputs
    } else if (detectedEngine == ONNX.name) {
      {

        var (encoderSession, encoderEnv): (OrtSession, OrtEnvironment) = (null, null)
        var (decoderSession, decoderEnv): (OrtSession, OrtEnvironment) = (null, null)

        val (_encoderSession, _encoderEnv) =
          onnxWrapper.get.encoder.getSession(onnxSessionOptions)
        val (_decoderSession, _decoderEnv) =
          onnxWrapper.get.decoder.getSession(onnxSessionOptions)

        encoderSession = _encoderSession
        encoderEnv = _encoderEnv
        decoderSession = _decoderSession
        decoderEnv = _decoderEnv

        val encoderAttentionMask: OnnxTensor =
          OnnxTensor.createTensor(
            encoderEnv,
            expandedEncoderInputIdsVals.toArray.map(_.map(_ => 1L)))

        val encoderInputTensors: OnnxTensor =
          OnnxTensor.createTensor(
            encoderEnv,
            expandedEncoderInputIdsVals.toArray.map(_.map(_.toLong)))

        val encoderInputs: java.util.Map[String, OnnxTensor] = Map(
          OnnxSignatures.encoderInputIDs -> encoderInputTensors,
          OnnxSignatures.encoderAttentionMask -> encoderAttentionMask).asJava

        val encoderResults = encoderSession.run(encoderInputs)

        val encoderStateBuffer =
          try {
            val encoderStateTensor = encoderResults
              .get(OnnxSignatures.encoderOutput)
              .get()
              .asInstanceOf[OnnxTensor]

            val shape = encoderStateTensor.getInfo.getShape
            encoderStateTensor.getFloatBuffer
              .array()
              .grouped(shape(2).toInt)
              .toArray
              .grouped(shape(1).toInt)
              .toArray
          } finally {
            if (encoderResults != null) encoderResults.close()
          }

        val decoderEncoderStateTensors = OnnxTensor.createTensor(encoderEnv, encoderStateBuffer)
        val modelOutputs = generate(
          batch,
          Right(decoderEncoderStateTensors),
          Right(encoderAttentionMask),
          decoderInputs,
          maxOutputLength,
          minOutputLength,
          doSample,
          beamSize,
          1,
          temperature,
          topK,
          topP,
          repetitionPenalty,
          noRepeatNgramSize,
          this.vocabSize,
          this.eosTokenId,
          this.paddingTokenId,
          randomSeed,
          ignoreTokenIdsInt,
          Right((decoderEnv, decoderSession)))

        encoderInputTensors.close()
        encoderAttentionMask.close()

        modelOutputs
      }

    } else {

      val encoderInferRequest =
        openvinoWrapper.get.encoder.getCompiledModel().create_infer_request()
      val decoderInferRequest =
        openvinoWrapper.get.decoder.getCompiledModel().create_infer_request()

      val encoderAttentionMask: org.intel.openvino.Tensor =
        new org.intel.openvino.Tensor(
          Array(expandedEncoderInputIdsVals.length, expandedEncoderInputIdsVals.head.length),
          expandedEncoderInputIdsVals.toArray.map(_.map(_ => 1L)).flatten)

      val encoderInputTensors =
        new org.intel.openvino.Tensor(
          Array(expandedEncoderInputIdsVals.length, expandedEncoderInputIdsVals.head.length),
          expandedEncoderInputIdsVals.toArray.map(_.map(_.toLong)).flatten)

      encoderInferRequest.set_tensor(OpenVinoSignatures.encoderInputIDs, encoderInputTensors)
      encoderInferRequest.set_tensor(
        OpenVinoSignatures.encoderAttentionMask,
        encoderAttentionMask)
      encoderInferRequest.infer()

      val encoderStateBuffer =
        try {
          val encoderStateTensor =
            encoderInferRequest.get_tensor(OpenVinoSignatures.encoderOutput)

          val shape = encoderStateTensor.get_shape().map(_.toLong)
          encoderStateTensor
            .data()
            .grouped(shape(2).toInt)
            .toArray
            .grouped(shape(1).toInt)
            .toArray
        } catch {
          case e: Exception =>
            e.printStackTrace()
            Array.empty[Float]
            // Rethrow the exception to propagate it further
            throw e
        }

      val decoderEncoderStateTensors =
        new org.intel.openvino.Tensor(
          Array(
            encoderStateBuffer.length,
            encoderStateBuffer.head.length,
            encoderStateBuffer.head.head.length),
          encoderStateBuffer.flatten.flatten)

      decoderEncoderStateTensorsOV = Some(decoderEncoderStateTensors)
      encoderAttentionMaskOV = Some(encoderAttentionMask)

      val modelOutputs = generate(
        batch,
        null,
        null,
        decoderInputs,
        maxOutputLength,
        minOutputLength,
        doSample,
        beamSize,
        1,
        temperature,
        topK,
        topP,
        repetitionPenalty,
        noRepeatNgramSize,
        this.vocabSize,
        this.eosTokenId,
        this.paddingTokenId,
        randomSeed,
        ignoreTokenIdsInt,
        null,
        ovInferRequest = Some(decoderInferRequest))

      modelOutputs

    }
  }

  /** Decode a sequence of sentences
    * @param sentences
    *   Sequence of sentences
    * @return
    *   Sequence of decoded sentences
    */
  def decode(sentences: Array[Array[Int]]): Seq[String] = {
    sentences.map(s => bpeTokenizer.decodeTokens(s.map(_.toInt)))
  }

  /** Encode a sequence of sentences
    * @param sentences
    *   Sequence of sentences
    * @param task
    *   Task
    * @return
    *   Sequence of encoded sentences
    */
  def encode(sentences: Seq[Annotation], task: String): Seq[Array[Int]] = {
    SentenceSplit
      .unpack(sentences)
      .map(s => {
        val sentWithTask =
          if (task.nonEmpty) s
          else s
        bpeTokenizer
          .tokenize(sentWithTask)
          .map(bpeTokenizer.encode)
          .flatMap(_.map(_.pieceId))
      })
  }

  /** Get model output for a batch of input sequences
    * @param encoderInputIds
    *   input ids
    * @param decoderInputIds
    *   decoder input ids
    * @param decoderEncoderStateTensors
    *   encoder state
    * @param encoderAttentionMaskTensors
    *   attention mask
    * @param maxLength
    *   max length
    * @param session
    *   tensorflow session
    * @return
    *   model output
    */
  override def getModelOutput(
      encoderInputIds: Seq[Array[Int]],
      decoderInputIds: Seq[Array[Int]],
      decoderEncoderStateTensors: Either[Tensor, OnnxTensor],
      encoderAttentionMaskTensors: Either[Tensor, OnnxTensor],
      maxLength: Int,
      session: Either[Session, (OrtEnvironment, OrtSession)],
      ovInferRequest: Option[InferRequest]): Array[Array[Float]] = {

    if (detectedEngine == TensorFlow.name) {
      // extract decoderEncoderStateTensors, encoderAttentionMaskTensors and Session from LEFT
      assert(decoderEncoderStateTensors.isLeft)
      assert(encoderAttentionMaskTensors.isLeft)
      assert(session.isLeft)

      val decoderEncoderStateTensor: Tensor = decoderEncoderStateTensors.left.get
      val encoderAttentionMaskTensor: Tensor = encoderAttentionMaskTensors.left.get
      val sess: Session = session.left.get

      val sequencesLength = encoderInputIds.map(x => x.length).toArray
      var maxSentenceLength = sequencesLength.max // - curLen
      maxSentenceLength = Math.max(maxSentenceLength, maxLength)
      val vocabSize = this.vocabSize
      val decoderInputLength = decoderInputIds.head.length
      val batchSize = encoderInputIds.length

      val useLastIdOnly = useCache && (decoderInputLength > 0)
      val sequenceLength = if (useLastIdOnly) 1 else decoderInputLength
      if (!useCache) {
        tensorDecoder = new TensorResources()
      }
      val decoderInputBuffers =
        tensorDecoder.createIntBuffer(decoderInputIds.length * sequenceLength)

      decoderInputIds.zipWithIndex.foreach { case (pieceIds, idx) =>
        val offset = idx * sequenceLength
        decoderInputBuffers
          .offset(offset)
          .write(if (useLastIdOnly) pieceIds.takeRight(1) else pieceIds)
      }

      val decoderInputTensors = tensorDecoder.createIntBufferTensor(
        Array(decoderInputIds.length, sequenceLength),
        decoderInputBuffers)

      val runner = if (nextStateTensor1.isEmpty || nextStateTensor2.isEmpty) {
        val r = sess.runner
          .feed(
            _tfBartSignatures.getOrElse(
              ModelSignatureConstants.InitDecoderInputIds.key,
              "missing_decoder_input_ids_init"),
            decoderInputTensors)
          .feed(
            _tfBartSignatures.getOrElse(
              ModelSignatureConstants.InitDecoderEncoderInputIds.key,
              "missing_encoder_state_init"),
            decoderEncoderStateTensor)
          .feed(
            _tfBartSignatures.getOrElse(
              ModelSignatureConstants.InitDecoderEncoderAttentionMask.key,
              "missing_decoder_encoder_attention_mask_init"),
            encoderAttentionMaskTensor)
          .fetch(_tfBartSignatures
            .getOrElse(ModelSignatureConstants.InitLogitsOutput.key, "missing_logits_init"))

        if (!useCache)
          r
        else
          r
            .fetch(
              _tfBartSignatures
                .getOrElse(
                  ModelSignatureConstants.InitCachedOutput1.key,
                  "missing_cache1_out_init"))
            .fetch(
              _tfBartSignatures
                .getOrElse(
                  ModelSignatureConstants.InitCachedOutPut2.key,
                  "missing_cache2_out_init"))
      } else {
        sess.runner
          .feed(
            _tfBartSignatures.getOrElse(
              ModelSignatureConstants.CachedDecoderInputIds.key,
              "missing_decoder_input_ids"),
            decoderInputTensors)
          .feed(
            _tfBartSignatures.getOrElse(
              ModelSignatureConstants.CachedDecoderEncoderInputIds.key,
              "missing_encoder_state"),
            decoderEncoderStateTensor)
          .feed(
            _tfBartSignatures.getOrElse(
              ModelSignatureConstants.CachedDecoderEncoderAttentionMask.key,
              "missing_decoder_encoder_attention_mask"),
            encoderAttentionMaskTensor)
          .feed(
            _tfBartSignatures.getOrElse(
              ModelSignatureConstants.CachedDecoderInputCache1.key,
              "missing_decoder_input_cache1"),
            nextStateTensor1.get)
          .feed(
            _tfBartSignatures.getOrElse(
              ModelSignatureConstants.CachedDecoderInputCache2.key,
              "missing_decoder_input_cache2"),
            nextStateTensor2.get)
          .fetch(_tfBartSignatures
            .getOrElse(ModelSignatureConstants.CachedLogitsOutput.key, "missing_logits_out"))
          .fetch(_tfBartSignatures
            .getOrElse(ModelSignatureConstants.CachedOutput1.key, "missing_cache1_out"))
          .fetch(_tfBartSignatures
            .getOrElse(ModelSignatureConstants.CachedOutPut2.key, "missing_cache2_out"))
      }

      val decoderOuts = runner.run().asScala
      val logitsRaw = TensorResources.extractFloats(decoderOuts.head)
      decoderOuts.head.close()
      val decoderOutputs = (0 until batchSize).map(i => {
        logitsRaw
          .slice(
            i * sequenceLength * vocabSize + (sequenceLength - 1) * vocabSize,
            i * sequenceLength * vocabSize + sequenceLength * vocabSize)
      })

      if (useCache) {
        if (nextStateTensor1.isDefined) {
          nextStateTensor1.get.close()
        }
        if (nextStateTensor2.isDefined) {
          nextStateTensor2.get.close()
        }
        nextStateTensor1 = Some(decoderOuts(1).asRawTensor())
        nextStateTensor2 = Some(decoderOuts(2).asRawTensor())
      }

      val nextTokenLogits = decoderOutputs.toArray
      if (!useCache) {
        tensorDecoder.clearSession(decoderOuts)
        tensorDecoder.clearTensors()
      }
      decoderInputTensors.close()
      nextTokenLogits
    } else if (detectedEngine == ONNX.name) {
      val (env, decoderSession) = session.right.get

      val decoderInputLength = decoderInputIds.head.length
      val sequenceLength = decoderInputLength
      val batchSize = encoderInputIds.length

      val decoderInputIdsLong: Array[Array[Long]] =
        decoderInputIds.map { tokenIds => tokenIds.map(_.toLong) }.toArray.map { tokenIds =>
          tokenIds
        }

      val decoderInputIdsLongTensor: OnnxTensor =
        OnnxTensor.createTensor(env, decoderInputIdsLong)

      val encoderAttentionMaskTensor = encoderAttentionMaskTensors.fold(
        tfTensor => {
          // not implemented yet
          null
        },
        onnxTensor => onnxTensor)

      val decoderEncoderStateTensor = decoderEncoderStateTensors.fold(
        tfTensor => {
          // not implemented yet
          null
        },
        onnxTensor => onnxTensor)

      val decoderInputs: java.util.Map[String, OnnxTensor] = Map(
        OnnxSignatures.decoderInputIDs -> decoderInputIdsLongTensor,
        OnnxSignatures.decoderEncoderAttentionMask -> encoderAttentionMaskTensor,
        OnnxSignatures.decoderEncoderState -> decoderEncoderStateTensor).asJava
      val sessionOutput = decoderSession.run(decoderInputs)

      val logitsRaw = sessionOutput.getFloatArray(OnnxSignatures.decoderOutput)
      val decoderOutputs = (0 until batchSize).map(i => {
        logitsRaw
          .slice(
            i * sequenceLength * vocabSize + (sequenceLength - 1) * vocabSize,
            i * sequenceLength * vocabSize + sequenceLength * vocabSize)
      })
      decoderOutputs.toArray

    } else {
      val decoderInputLength = decoderInputIds.head.length
      val sequenceLength = decoderInputLength
      val batchSize = encoderInputIds.length

      val decoderInputIdsLong: Array[Array[Long]] =
        decoderInputIds.map { tokenIds => tokenIds.map(_.toLong) }.toArray.map { tokenIds =>
          tokenIds
        }

      val decoderInputIdsLongTensor =
        new org.intel.openvino.Tensor(
          Array(decoderInputIdsLong.length, decoderInputIdsLong.head.length),
          decoderInputIdsLong.flatten)

      ovInferRequest.get.set_tensor(OpenVinoSignatures.decoderInputIDs, decoderInputIdsLongTensor)
      ovInferRequest.get.set_tensor(
        OpenVinoSignatures.decoderEncoderAttentionMask,
        encoderAttentionMaskOV.get)
      ovInferRequest.get.set_tensor(
        OpenVinoSignatures.decoderEncoderState,
        decoderEncoderStateTensorsOV.get)

      ovInferRequest.get.infer()

      val logitsRaw = ovInferRequest.get.get_tensor(OpenVinoSignatures.decoderOutput).data()
      val decoderOutputs = (0 until batchSize).map(i => {
        logitsRaw
          .slice(
            i * sequenceLength * vocabSize + (sequenceLength - 1) * vocabSize,
            i * sequenceLength * vocabSize + sequenceLength * vocabSize)
      })
      decoderOutputs.toArray

    }
  }

  private def sessionWarmup(): Unit = {
    val dummyInput = Array.fill(1)(0) ++ Array(eosTokenId)
    tag(
      Seq(dummyInput),
      minOutputLength = 0,
      maxOutputLength = 1,
      doSample = false,
      temperature = 0f,
      topK = 0,
      topP = 0f,
      repetitionPenalty = 0f,
      noRepeatNgramSize = 0,
      randomSeed = Option(0),
      ignoreTokenIds = Array(0),
      beamSize = 1,
      maxInputLength = 512)

  }
}
