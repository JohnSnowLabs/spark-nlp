/*
 * Copyright 2017 - 2024  John Snow Labs
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
import com.johnsnowlabs.ml.ai.util.Generation.Logit.LogitProcess.{NoRepeatNgramsLogitProcessor, RepetitionPenaltyLogitProcessor}
import com.johnsnowlabs.ml.ai.util.Generation.Logit.LogitProcessorList
import com.johnsnowlabs.ml.ai.util.Generation.Logit.LogitWarper.{TemperatureLogitWarper, TopKLogitWarper, TopPLogitWarper}
import com.johnsnowlabs.ml.ai.util.Generation.Search.{BeamScorer, BeamSearchScorer}
import com.johnsnowlabs.ml.tensorflow.{TensorResources, TensorflowWrapper}
import com.johnsnowlabs.ml.tensorflow.sign.{ModelSignatureConstants, ModelSignatureManager}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import com.johnsnowlabs.nlp.annotators.common.SentenceSplit
import com.johnsnowlabs.nlp.annotators.tokenizer.bpe.{BlenderBotTokenizer, BpeTokenizer}
import org.intel.openvino.InferRequest
import org.tensorflow.{Session, Tensor}
import org.tensorflow.ndarray.buffer.IntDataBuffer

import scala.collection.JavaConverters._
import scala.util.control.Breaks.{break, breakable}

private[johnsnowlabs] class BlenderBotGenerate(
  val tensorflow: TensorflowWrapper,
  configProtoBytes: Option[Array[Byte]] = None,
  signatures: Option[Map[String, String]] = None,
  merges: Map[(String, String), Int],
  vocabulary: Map[String, Int],
  useCache: Boolean = false)
  extends Serializable
  with Generate {

  val bpeTokenizer: BlenderBotTokenizer = BpeTokenizer
    .forModel("blenderbot", merges = merges, vocab = vocabulary)
    .asInstanceOf[BlenderBotTokenizer]

  private val _tfBlenderBotSignatures: Map[String, String] =
    signatures.getOrElse(ModelSignatureManager.apply())

  private val paddingTokenId = 1
  private val eosTokenId = 2
  private val vocabSize = 8008
  var tensorDecoder = new TensorResources()
  private var nextStateTensor1: Option[org.tensorflow.Tensor] = None
  private var nextStateTensor2: Option[org.tensorflow.Tensor] = None

  def predict(
    sentences: Seq[Annotation],
    batchSize: Int,
    maxOutputLength: Int,
    doSample: Boolean,
    temperature: Double,
    topK: Int,
    topP: Double,
    repetitionPenalty: Double,
    noRepeatNgramSize: Int,
    task: String,
    randomSeed: Option[Long] = None,
    beamSize: Int,
    maxInputLength: Int): Seq[Annotation] = {
    val batchDecoder = sentences.grouped(batchSize).toArray.flatMap { batch =>
      val batchSP = encode(batch, task)
      val spIds = tag(
        batchSP,
        maxOutputLength,
        doSample,
        temperature,
        topK,
        topP,
        repetitionPenalty,
        noRepeatNgramSize,
        randomSeed,
        beamSize,
        maxInputLength)
      val result = decode(spIds)
      result
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

  def tag(
    batch: Seq[Array[Int]],
    maxOutputLength: Int,
    doSample: Boolean,
    temperature: Double,
    topK: Int,
    topP: Double,
    repetitionPenalty: Double,
    noRepeatNgramSize: Int,
    randomSeed: Option[Long],
    beamSize: Int,
    maxInputLength: Int): Array[Array[Int]] = {
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

    val tokenBuffers: IntDataBuffer = tensorEncoder.createIntBuffer(inputDim)
    val tokenTensors = tensorEncoder.createIntBufferTensor(shape, tokenBuffers)

    val session = tensorflow.getTFSessionWithSignature(
      configProtoBytes = configProtoBytes,
      initAllTables = false,
      savedSignatures = signatures)

    val decoderInputTensors = tensorEncoder.createIntBufferTensor(shape, encoderInputBuffers)
    val encoderAttentionMaskTensors =
      tensorEncoder.createIntBufferTensor(shape, encoderAttentionMaskBuffers)

    val runner = session.runner
      .feed(
        _tfBlenderBotSignatures.getOrElse(ModelSignatureConstants.InputIds.key, "missing_input_id_key"),
        tokenTensors)
      .feed(
        _tfBlenderBotSignatures.getOrElse(ModelSignatureConstants.AttentionMask.key, "missing_input_mask_key"),
        encoderAttentionMaskTensors
      )
      .feed(
        _tfBlenderBotSignatures.getOrElse(ModelSignatureConstants.DecoderInputIds.key, "missing_decoder_input_ids"),
        decoderInputTensors
      )
      .fetch(
        _tfBlenderBotSignatures.getOrElse("encoder_last_hidden_state", "missing_last_hidden_state")
      )
      .fetch(_tfBlenderBotSignatures
        .getOrElse(ModelSignatureConstants.LogitsOutput.key, "missing_logits_init"))

    val outs = runner.run().asScala
    val encoderOutsFloats = TensorResources.extractFloats(outs.head)
    val logits = TensorResources.extractFloats(outs.last)

    val dim = encoderOutsFloats.length / inputDim
    val encoderOutsBatch =
      encoderOutsFloats.grouped(dim).toArray.grouped(maxSentenceLength).toArray

    outs.foreach(_.close())

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
    val decoderInputs = batch.map(_ => Array(this.eosTokenId)).toArray

    val modelOutputs = generateBlenderBot(
      inputIds = batch,
      decoderInputs = decoderInputs,
      logits = logits,
      doSample = doSample,
      maxOutputLength = maxOutputLength,
      beamSize = beamSize,
      numReturnSequences = 1,
      repetitionPenalty = repetitionPenalty,
      noRepeatNgramSize = noRepeatNgramSize,
      temperature = temperature,
      topK = topK,
      topP = topP,
      eosTokenId = this.eosTokenId,
      paddingTokenId = this.paddingTokenId,
      randomSeed = randomSeed)

    tensorEncoder.clearTensors()
    tensorEncoder.clearSession(outs)
    decoderEncoderStateTensorResources.clearTensors()
    decoderEncoderStateTensors.close()
    encoderAttentionMaskTensors.close()
    tokenTensors.close()
    if (useCache) {
      tensorDecoder.clearTensors()
      nextStateTensor1 = None
      nextStateTensor2 = None
    }
    modelOutputs
  }

  def generateBlenderBot(
    inputIds: Seq[Array[Int]],
    decoderInputs: Array[Array[Int]],
    logits: Array[Float],
    doSample: Boolean,
    maxOutputLength: Int,
    beamSize: Int,
    numReturnSequences: Int,
    repetitionPenalty: Double,
    noRepeatNgramSize: Int,
    temperature: Double,
    topK: Int,
    topP: Double,
    eosTokenId: Int,
    paddingTokenId: Int,
    randomSeed: Option[Long],
    applySoftmax: Boolean = true
  ): Array[Array[Int]] = {
    val logitProcessorList = new LogitProcessorList()

    logitProcessorList.addProcess(new RepetitionPenaltyLogitProcessor(repetitionPenalty))

    logitProcessorList.addProcess(
      new NoRepeatNgramsLogitProcessor(
        noRepeatNgramSize = noRepeatNgramSize,
        vocabSize = vocabSize))

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

    beamSearchBlenderBot(
      inputIds,
      decoderInputs,
      beamSearchScorer,
      logits,
      applySoftmax,
      logitProcessorList,
      maxOutputLength,
      paddingTokenId,
      eosTokenId,
      doSample,
      randomSeed
    )
  }

  def beamSearchBlenderBot(
    encoderInputIdsVals: Seq[Array[Int]],
    inputIdsVal: Seq[Array[Int]],
    beamScorer: BeamScorer,
    logits: Array[Float],
    applySoftmax: Boolean,
    logitProcessor: LogitProcessorList,
    maxLength: Int,
    padTokenId: Int,
    eosTokenId: Int,
    doSample: Boolean,
    randomSeed: Option[Long],
    stopTokenIds: Array[Int] = Array()): Array[Array[Int]] = {
    val inputIds = inputIdsVal
    val batchSize = beamScorer.getBeamHypothesesSeq.length
    val numBeams = beamScorer.getNumBeams
    val batchBeamSize = batchSize * numBeams
    var currentLength = inputIds.head.length

    var beamScores = Array.ofDim[Float](batchSize * numBeams)
    beamScores = beamScores.zipWithIndex.map { case (_, ind) =>
      if (ind % numBeams == 0) 0 else (-1e+9).toFloat
    }
    var beamIndices = Seq.fill(batchBeamSize)(Array[Int]())
    var nextIndices = Array[Array[Int]]()
    var nextTokens = Array[Array[Int]]()
    var expandedInputs = inputIds.flatMap(x => List.fill(numBeams)(x))
    val expandedEncoderInputIdsVals = encoderInputIdsVals.flatMap(x => List.fill(numBeams)(x))
    breakable {
      while (true) {
        val nextTokenLogits = getModelOutputBlenderBot(
          expandedEncoderInputIdsVals,
          expandedInputs,
          logits)

        // Optionally Apply log softmax to model outputs
        var nextTokenScores =
          if (applySoftmax) nextTokenLogits.map(logSoftmax) else nextTokenLogits
        // Process the logits by defined logit processors
        val nextTokenScoresProcessed =
          logitProcessor.process(expandedInputs, nextTokenScores, currentLength)

        // Process the logits by defined logit warpers
        if (doSample) {
          nextTokenScores =
            logitProcessor.warp(expandedInputs, nextTokenScoresProcessed, currentLength)
        }
        // Add previous beam scores to the output
        nextTokenScores = nextTokenScores.zipWithIndex.map { case (x, ind1) =>
          x.zipWithIndex.map { case (y, _) =>
            y + beamScores(ind1)
          }
        }

        // Reshape next token score to (batchSize, vocabSize * numBeams)
        val vocabSize = nextTokenScores.head.length
        val reshapedNextTokenScores =
          reshapeArray(nextTokenScores, batchSize, vocabSize * numBeams)

        nextTokenScores = reshapedNextTokenScores

        var nextKTopTokenScores: Array[Array[Float]] = Array[Array[Float]]()
        var nextKTopTokens: Array[Array[Int]] = Array[Array[Int]]()

        if (doSample) {
          val nextKIndices = nextTokenScores.map(x => {
            multinomialSampling(x, 2 * numBeams, randomSeed)
          })
          nextKTopTokenScores = Array.ofDim[Float](nextKIndices.length, nextKIndices.head.length)
          for (i <- nextKIndices.indices) {
            for (j <- nextKIndices(i).indices) {
              nextKTopTokenScores(i)(j) = nextTokenScores(i)(nextKIndices(i)(j))
            }
          }
          nextKTopTokenScores =
            nextKTopTokenScores.map(x => x.zipWithIndex.sortWith(_._1 > _._1).map(_._1))
          val tempNextKInd =
            nextKTopTokenScores.map(x => x.zipWithIndex.sortWith(_._1 > _._1).map(_._2))
          nextKTopTokens = Array.ofDim[Int](nextKIndices.length, nextKIndices.head.length)

          for (i <- tempNextKInd.indices) {
            for (j <- tempNextKInd(i).indices) {
              nextKTopTokens(i)(j) = nextKIndices(i)(tempNextKInd(i)(j))
            }
          }
        } else {
          nextKTopTokenScores = nextTokenScores.map(x =>
            x.zipWithIndex.sortWith(_._1 > _._1).take(2 * numBeams).map(_._1))
          nextKTopTokens = nextTokenScores.map(x =>
            x.zipWithIndex.sortWith(_._1 > _._1).take(2 * numBeams).map(_._2))
        }
        nextIndices = nextKTopTokens.map(y => y.map(x => x / vocabSize))
        nextTokens = nextKTopTokens.map(y => y.map(x => x % vocabSize))

        val beamOutputs = beamScorer.process(
          expandedInputs,
          nextKTopTokenScores,
          nextTokens,
          nextIndices,
          padTokenId,
          eosTokenId,
          beamIndices,
          currentLength,
          stopTokenIds)
        val newBeamScores = beamOutputs._1.flatMap(_.toList)
        val beamNextTokens = beamOutputs._2.flatMap(_.toList)
        val beamIdx = beamOutputs._3.flatMap(_.toList)
        var newInputIds = Seq[Array[Int]]()
        for ((i, ind) <- beamIdx.zipWithIndex) {
          val tempInput = expandedInputs(i) :+ beamNextTokens(ind)
          newInputIds = newInputIds :+ tempInput
        }
        expandedInputs = newInputIds
        beamScores = newBeamScores
        beamIndices = beamIndices.indices.map { elem =>
          beamIndices(beamIdx(elem)) :+ beamIdx(elem)
        }
        currentLength = currentLength + 1
        if (beamScorer.isDone || (expandedInputs.head.length >= maxLength)) {
          break
        }
      }
    }

    val sequenceOutputs = beamScorer.finalize(
      inputIds = expandedInputs,
      finalBeamScores = beamScores,
      finalBeamTokens = nextTokens.flatMap(_.toList),
      finalBeamIndices = nextIndices.flatMap(_.toList),
      maxLength = maxLength,
      padTokenId = padTokenId,
      eosTokenId = eosTokenId,
      beamIndices = beamIndices)

    sequenceOutputs._1
  }

  def getModelOutputBlenderBot(
    encoderInputIds: Seq[Array[Int]],
    decoderInputIds: Seq[Array[Int]],
    logitsRaw: Array[Float]): Array[Array[Float]] = {

    val batchSize = encoderInputIds.length
    val decoderInputLength = decoderInputIds.head.length
    val useLastIdOnly = useCache && (decoderInputLength > 0)
    val sequenceLength = if (useLastIdOnly) 1 else decoderInputLength

    val decoderOutputs = (0 until batchSize).map(i => {
      logitsRaw
        .slice(
          i * sequenceLength * vocabSize + (sequenceLength - 1) * vocabSize,
          i * sequenceLength * vocabSize + sequenceLength * vocabSize)
    })

    val nextTokenLogits = decoderOutputs.toArray
    nextTokenLogits
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

  /** Calls the model and returns the output logits.
   *
   * @param encoderInputIds
   * Input IDs for the Encoder
   * @param decoderInputIds
   * Input IDs for the Decoder
   * @param decoderEncoderStateTensors
   * Tensor of encoded input for the decoder
   * @param encoderAttentionMaskTensors
   * Tensor for encoder attention mask
   * @param maxLength
   * Max length of the input
   * @param session
   * Tensorflow Session
   * @return
   * Logits for the input
   */
  override def getModelOutput(encoderInputIds: Seq[Array[Int]], decoderInputIds: Seq[Array[Int]], decoderEncoderStateTensors: Either[Tensor, OnnxTensor], encoderAttentionMaskTensors: Either[Tensor, OnnxTensor], maxLength: Int, session: Either[Session, (OrtEnvironment, OrtSession)], ovInferRequest: Option[InferRequest]): Array[Array[Float]] = {
    Array()
  }
}
