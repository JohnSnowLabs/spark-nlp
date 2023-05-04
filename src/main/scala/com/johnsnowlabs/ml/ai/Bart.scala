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

import com.johnsnowlabs.ml.ai.util.Generation.Generate
import com.johnsnowlabs.ml.ai.util.Generation.Logit.LogitProcess.{
  MinLengthLogitProcessor,
  NoRepeatNgramsLogitProcessor,
  RepetitionPenaltyLogitProcessor
}
import com.johnsnowlabs.ml.ai.util.Generation.Logit.LogitProcessorList
import com.johnsnowlabs.ml.ai.util.Generation.Logit.LogitWarper.{
  TemperatureLogitWarper,
  TopKLogitWarper,
  TopPLogitWarper
}
import com.johnsnowlabs.ml.ai.util.Generation.Search.BeamSearchScorer
import com.johnsnowlabs.ml.tensorflow.sign.{ModelSignatureConstants, ModelSignatureManager}
import com.johnsnowlabs.ml.tensorflow.{TensorResources, TensorflowWrapper}
import com.johnsnowlabs.nlp.annotators.tokenizer.bpe.{BpeTokenizer, BartTokenizer}
import com.johnsnowlabs.nlp.annotators.common.SentenceSplit
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import org.tensorflow.{Session, Tensor}

import scala.collection.JavaConverters._
import scala.collection.mutable

/** This class is used to run Bart model for For Sequence Batches of WordpieceTokenizedSentence.
  * Input for this model must be tokenized with a SentencePieceModel,
  *
  * @param tensorflow
  *   BART Model wrapper with TensorFlowWrapper
  * @param bpeTokenizer
  *   BART Byte-Pair Encoder model with BPEWrapper
  * @param configProtoBytes
  *   Configuration for TensorFlow session
  */

private[johnsnowlabs] class Bart(
    val tensorflow: TensorflowWrapper,
    configProtoBytes: Option[Array[Byte]] = None,
    signatures: Option[Map[String, String]] = None,
    merges: Map[(String, String), Int],
    vocabulary: Map[String, Int],
    useCache: Boolean = false)
    extends Serializable
    with Generate {

  private val _tfBartSignatures: Map[String, String] =
    signatures.getOrElse(ModelSignatureManager.apply())

  val bpeTokenizer: BartTokenizer = BpeTokenizer
    .forModel("bart", merges = merges, vocab = vocabulary, padWithSentenceTokens = false)
    .asInstanceOf[BartTokenizer]

  private val paddingTokenId = 1
  private val eosTokenId = 2
  private val vocab_size = 50264
  private val encoderInputIdsKey = "encoder_encoder_input_ids:0"
  private val encoderAttentionMaskKey = "encoder_encoder_attention_mask:0"
  private val encoderOutputKey = "StatefulPartitionedCall_2:0"

  private val decoderInitInputIdsKey = "decoder_init_decoder_input_ids:0"
  private val decoderInitEncoderAttentionMaskKey = "decoder_init_encoder_attention_mask:0"
  private val decoderInitEncoderStateKey = "decoder_init_encoder_state:0"

  private val decoderInitOutputLogitsKey = "StatefulPartitionedCall_1:2"
  private val decoderInitOutputCache1Key = "StatefulPartitionedCall_1:0"
  private val decoderInitOutputCache2Key = "StatefulPartitionedCall_1:1"

  private val decoderCachedInputIdsKey = "decoder_cached_decoder_input_ids:0"
  private val decoderCachedEncoderAttentionMaskKey = "decoder_cached_encoder_attention_mask:0"
  private val decoderCachedEncoderStateKey = "decoder_cached_encoder_state:0"
  private val decoderCachedCache1Key = "decoder_cached_cache1:0"
  private val decoderCachedCache2Key = "decoder_cached_cache2:0"

  private val decoderCachedOutputLogitsKey = "StatefulPartitionedCall:2"
  private val decoderCachedOutputCache1Key = "StatefulPartitionedCall:0"
  private val decoderCachedOutputCache2Key = "StatefulPartitionedCall:1"
  private var nextStateTensor1: Option[org.tensorflow.Tensor] = None
  private var nextStateTensor2: Option[org.tensorflow.Tensor] = None
  var tensorDecoder = new TensorResources()

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
      beamSize = 1)
  }

//  sessionWarmup()

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
      beamSize: Int): Array[Array[Int]] = {
    val ignoreTokenIdsInt = ignoreTokenIds
    val expandedEncoderInputIdsVals = batch.flatMap(x => List.fill(beamSize)(x))
//    val expandedEncoderInputIdsVals = batch
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

    val session = tensorflow.getTFSessionWithSignature(
      configProtoBytes = configProtoBytes,
      initAllTables = false,
      savedSignatures = signatures)

    val encoderInputTensors = tensorEncoder.createIntBufferTensor(shape, encoderInputBuffers)
    val encoderAttentionMaskTensors =
      tensorEncoder.createIntBufferTensor(shape, encoderAttentionMaskBuffers)

    val runner = session.runner;

    runner
      .feed(encoderInputIdsKey, encoderInputTensors)
      .feed(encoderAttentionMaskKey, encoderAttentionMaskTensors)
      .fetch(encoderOutputKey)

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

    val modelOutputs = generateBeamSearch(
      batch,
      decoderEncoderStateTensors,
      encoderAttentionMaskTensors,
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
      randomSeed,
      ignoreTokenIdsInt,
      session)

    tensorEncoder.clearTensors()
    tensorEncoder.clearSession(encoderOuts)
    decoderEncoderStateTensorResources.clearTensors()
    decoderEncoderStateTensors.close()
    encoderAttentionMaskTensors.close()
    encoderInputTensors.close()
    if (useCache) {
      tensorDecoder.clearTensors()
      nextStateTensor1 match {
        case Some(t) => t.close()
        case None =>
      }
      nextStateTensor2 match {
        case Some(t) => t.close()
        case None =>
      }
    }
    modelOutputs
  }

  def generateBeamSearch(
      inputIds: Seq[Array[Int]],
      decoderEncoderStateTensors: Tensor,
      encoderAttentionMaskTensors: Tensor,
      maxOutputLength: Int,
      minOutputLength: Int,
      doSample: Boolean,
      beamSize: Int,
      numReturnSequences: Int,
      temperature: Double,
      topK: Int,
      topP: Double,
      repetitionPenalty: Double,
      noRepeatNgramSize: Int,
      randomSeed: Option[Long],
      ignoreTokenIds: Array[Int] = Array(),
      session: Session): Array[Array[Int]] = {

    var decoderInputs = inputIds.map(_ => Array(this.eosTokenId)).toArray

    var logitProcessorList = new LogitProcessorList()

    logitProcessorList.addProcess(new RepetitionPenaltyLogitProcessor(repetitionPenalty))

    logitProcessorList.addProcess(
      new NoRepeatNgramsLogitProcessor(
        noRepeatNgramSize = noRepeatNgramSize,
        vocabSize = this.vocab_size))

    logitProcessorList.addProcess(
      new MinLengthLogitProcessor(this.eosTokenId, minOutputLength, this.vocab_size))

    logitProcessorList.addProcess(new TemperatureLogitWarper(temperature))

    logitProcessorList.addProcess(new TopKLogitWarper(topK))

    logitProcessorList.addProcess(new TopPLogitWarper(topP))

    val beamSearchScorer = new BeamSearchScorer(
      beamSize = beamSize,
      batchSize = inputIds.length,
      lengthPenalty = repetitionPenalty.toFloat,
      doEarlyStopping = false,
      numBeamHypothesisToKeep = numReturnSequences,
      maxLength = maxOutputLength)

    this.beamSearch(
      inputIds,
      decoderInputs,
      decoderEncoderStateTensors,
      encoderAttentionMaskTensors,
      beamSearchScorer,
      logitProcessorList,
      maxOutputLength,
      this.paddingTokenId,
      this.eosTokenId,
      doSample,
      randomSeed,
      session)
  }

  def decode(sentences: Array[Array[Int]]): Seq[String] = {
    sentences.map(s => bpeTokenizer.decodeTokens(s.map(_.toInt)))
  }

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
      beamSize: Int): Seq[Annotation] = {

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
        beamSize)

      decode(spIds)

    }

    var sentBegin, nextSentEnd = 0
    batchDecoder.zip(sentences).map { case (content, sent) =>
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
  }

  override def getModelOutput(
      encoderInputIds: Seq[Array[Int]],
      decoderInputIds: Seq[Array[Int]],
      decoderEncoderStateTensors: Tensor,
      encoderAttentionMaskTensors: Tensor,
      maxLength: Int,
      session: Session): Array[Array[Float]] = {

    val sequencesLength = encoderInputIds.map(x => x.length).toArray
    var maxSentenceLength = sequencesLength.max // - curLen
    maxSentenceLength = Math.max(maxSentenceLength, maxLength)
    val vocabSize = this.vocab_size
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
      val r = session.runner
        .feed(decoderInitInputIdsKey, decoderInputTensors)
        .feed(decoderInitEncoderStateKey, decoderEncoderStateTensors)
        .feed(decoderInitEncoderAttentionMaskKey, encoderAttentionMaskTensors)
        .fetch(decoderInitOutputLogitsKey)

      if (!useCache)
        r
      else
        r
          .fetch(decoderInitOutputCache1Key)
          .fetch(decoderInitOutputCache2Key)
    } else {
      session.runner
        .feed(decoderCachedInputIdsKey, decoderInputTensors)
        .feed(decoderCachedEncoderStateKey, decoderEncoderStateTensors)
        .feed(decoderCachedEncoderAttentionMaskKey, encoderAttentionMaskTensors)
        .feed(decoderCachedCache1Key, nextStateTensor1.get)
        .feed(decoderCachedCache2Key, nextStateTensor2.get)
        .fetch(decoderCachedOutputLogitsKey)
        .fetch(decoderCachedOutputCache1Key)
        .fetch(decoderCachedOutputCache2Key)
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
  }
}
