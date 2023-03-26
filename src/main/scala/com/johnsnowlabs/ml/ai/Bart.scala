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
import com.johnsnowlabs.ml.tensorflow.sign.ModelSignatureManager
import com.johnsnowlabs.ml.tensorflow.{TensorResources, TensorflowWrapper}
import com.johnsnowlabs.nlp.annotators.common.SentenceSplit
import com.johnsnowlabs.nlp.annotators.tokenizer.bpe.BartTokenizer
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import com.johnsnowlabs.nlp.annotators.common.{Sentence, SentenceSplit}
import com.johnsnowlabs.ml.tensorflow.sign.{ModelSignatureConstants, ModelSignatureManager}
import org.tensorflow.{Session, Tensor}

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.math._

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
    val bpeTokenizer: BartTokenizer,
    configProtoBytes: Option[Array[Byte]] = None,
    signatures: Option[Map[String, String]] = None)
    extends Serializable
    with Generate {

  private val _tfBartSignatures: Map[String, String] =
    signatures.getOrElse(ModelSignatureManager.apply())
  private val paddingTokenId = 1L
  private val eosTokenId = 2L
  private val vocab_size = 50264
  var decoderEncoderStateTensors: Tensor =
    new TensorResources().createTensor(Array(1.0f, 2.0f, 3.0f))
  var encoderAttentionMaskTensors: Tensor =
    new TensorResources().createTensor(Array(1.0f, 2.0f, 3.0f))
  private val session = tensorflow.getTFSessionWithSignature(
    configProtoBytes = configProtoBytes,
    initAllTables = false,
    savedSignatures = signatures)

  private def sessionWarmup(): Unit = {
    val dummyInput = Array.fill(1)(0L) ++ Array(eosTokenId)
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
      randomSeed = Option(0L),
      ignoreTokenIds = Array(0),
      beamSize = 1)
  }

  sessionWarmup()

  def tag(
      batch: Seq[Array[Long]],
      minOutputLength: Int,
      maxOutputLength: Int,
      doSample: Boolean,
      temperature: Double,
      topK: Int,
      topP: Double,
      repetitionPenalty: Double,
      noRepeatNgramSize: Int,
      randomSeed: Option[Long],
      ignoreTokenIds: Array[Long] = Array(),
      beamSize: Int): Array[Array[Long]] = {

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

    val encoderInputBuffers = tensorEncoder.createLongBuffer(inputDim)
    val encoderAttentionMaskBuffers = tensorEncoder.createLongBuffer(inputDim)

    val shape = Array(expandedEncoderInputIdsVals.length.toLong, maxSentenceLength)

    expandedEncoderInputIdsVals.zipWithIndex.foreach { case (tokenIds, idx) =>
      val offset = idx * maxSentenceLength
      val diff = maxSentenceLength - tokenIds.length

      val s = tokenIds.take(maxSentenceLength) ++ Array.fill[Long](diff)(this.paddingTokenId)
      encoderInputBuffers.offset(offset).write(s)
      val mask = s.map(x => if (x != this.paddingTokenId) 1L else 0L)
      encoderAttentionMaskBuffers.offset(offset).write(mask)
    }

    val encoderInputTensors = tensorEncoder.createLongBufferTensor(shape, encoderInputBuffers)
    this.encoderAttentionMaskTensors =
      tensorEncoder.createLongBufferTensor(shape, encoderAttentionMaskBuffers)

    val runner = this.session.runner
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
        this.encoderAttentionMaskTensors)
      .fetch(_tfBartSignatures
        .getOrElse(ModelSignatureConstants.EncoderOutput.key, "missing_last_hidden_state"))

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

    this.decoderEncoderStateTensors = tensorEncoder.createFloatBufferTensor(
      Array(expandedEncoderInputIdsVals.length.toLong, maxSentenceLength, dim),
      decoderEncoderStateBuffers)

//    val modelOutputs = generateNoBeamSearch(
//      batch,
//      maxOutputLength,
//      minOutputLength,
//      doSample,
//      temperature,
//      topK,
//      topP,
//      repetitionPenalty,
//      noRepeatNgramSize,
//      randomSeed,
//      ignoreTokenIds)

    val modelOutputs = generateBeamSearch(
      batch,
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
      ignoreTokenIds)

    tensorEncoder.clearTensors()
    tensorEncoder.clearSession(encoderOuts)
    modelOutputs

  }

  def generateBeamSearch(
      inputIds: Seq[Array[Long]],
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
      ignoreTokenIds: Array[Long] = Array()): Array[Array[Long]] = {

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

    var beamSearchScorer = new BeamSearchScorer(
      beamSize = beamSize,
      batchSize = inputIds.length,
      lengthPenalty = repetitionPenalty.toFloat,
      doEarlyStopping = false,
      numBeamHypothesisToKeep = numReturnSequences,
      maxLength = maxOutputLength)

    this.beamSearch(
      inputIds,
      decoderInputs,
      beamSearchScorer,
      logitProcessorList,
      maxOutputLength,
      this.paddingTokenId,
      this.eosTokenId,
      doSample,
      randomSeed)
  }

  def generateNoBeamSearch(
      inputIds: Seq[Array[Long]],
      maxOutputLength: Int,
      minOutputLength: Int,
      doSample: Boolean,
      temperature: Double,
      topK: Int,
      topP: Double,
      repetitionPenalty: Double,
      noRepeatNgramSize: Int,
      randomSeed: Option[Long],
      ignoreTokenIds: Array[Long] = Array()): Array[Array[Long]] = {

    /** Generate sequences for each example without beam search (numBeams == 1). All returned
      * sequence are generated independently.
      */

    /* Actual size of each sentence to skip padding in the TF model */
    val sequencesLength = inputIds.map(x => x.length).toArray
    val maxSentenceLength = sequencesLength.max // - curLen

    val numReturn_sequences = 1
    // from config
    var effectiveBatch_size = 1
    var effectiveBatch_mult = 1

    // set effective batch size and effective batch multiplier according to do_sample
    if (doSample) {
      effectiveBatch_size = inputIds.length * numReturn_sequences
      effectiveBatch_mult = numReturn_sequences
    } else {
      effectiveBatch_size = inputIds.length
      effectiveBatch_mult = 1
    }

    // Run encoder
//    val tensorEncoder = new TensorResources()
//    val inputDim = inputIds.length * maxSentenceLength
//
//    val encoderInputBuffers = tensorEncoder.createLongBuffer(inputDim)
//    val encoderAttentionMaskBuffers = tensorEncoder.createLongBuffer(inputDim)
//
//    val shape = Array(inputIds.length.toLong, maxSentenceLength)
//
//    inputIds.zipWithIndex.foreach { case (tokenIds, idx) =>
//      val offset = idx * maxSentenceLength
//      val diff = maxSentenceLength - tokenIds.length
//
//      val s = tokenIds.take(maxSentenceLength) ++ Array.fill[Int](diff)(this.paddingTokenId)
//      encoderInputBuffers.offset(offset).write(s)
//      val mask = s.map(x => if (x != this.paddingTokenId) 1 else 0)
//      encoderAttentionMaskBuffers.offset(offset).write(mask)
//    }
//
//    val encoderInputTensors = tensorEncoder.createLongBufferTensor(shape, encoderInputBuffers)
//    val encoderAttentionMaskTensors =
//      tensorEncoder.createLongBufferTensor(shape, encoderAttentionMaskBuffers)
//
//    val session = tensorflow.getTFSessionWithSignature(
//      configProtoBytes = configProtoBytes,
//      initAllTables = false,
//      savedSignatures = signatures)

    var decoderInputs = inputIds.map(_ => Array(this.eosTokenId)).toArray

    val batch_size = effectiveBatch_size
    var curLen = decoderInputs(0).length

    var stopDecoder = false

    // length of generated sentences / unfinished sentences
    var unfinishedSents = List.fill(decoderInputs.length)(1)
    var sentLengths = List.fill(decoderInputs.length)(maxOutputLength)

    while (!stopDecoder) {
      val decoderInputLength = decoderInputs.head.length
      val tensorDecoder = new TensorResources()

      val decoderInputBuffers =
        tensorDecoder.createLongBuffer(decoderInputs.length * decoderInputLength)
      val decoderAttentionBuffers =
        tensorDecoder.createLongBuffer(decoderInputs.length * decoderInputLength)

      decoderInputs.zipWithIndex.foreach { case (pieceIds, idx) =>
        val offset = idx * decoderInputLength
        decoderInputBuffers.offset(offset).write(pieceIds)
        val paddingMasks = pieceIds.map(_ => 1L)
        decoderAttentionBuffers.offset(offset).write(paddingMasks)
      }

      val decoderInputTensors = tensorDecoder.createLongBufferTensor(
        Array(decoderInputs.length.toLong, decoderInputLength),
        decoderInputBuffers)
      val decoderAttentionMaskTensors = tensorDecoder.createLongBufferTensor(
        Array(decoderInputs.length.toLong, decoderInputLength),
        decoderAttentionBuffers)

      val runner = this.session.runner

      // TODO add past to the model and use cache
      runner
        .feed(
          _tfBartSignatures.getOrElse(
            ModelSignatureConstants.DecoderInputIds.key,
            "missing_decoder_input_ids"),
          decoderInputTensors)
        .feed(
          _tfBartSignatures.getOrElse(
            ModelSignatureConstants.DecoderAttentionMask.key,
            "missing_encoder_attention_mask"),
          decoderAttentionMaskTensors)
        .feed(
          _tfBartSignatures.getOrElse(
            ModelSignatureConstants.DecoderEncoderInputIds.key,
            "missing_encoder_state"),
          decoderEncoderStateTensors)
        .feed(
          _tfBartSignatures.getOrElse(
            ModelSignatureConstants.DecoderEncoderAttentionMask.key,
            "missing_decoder_encoder_attention_mask"),
          encoderAttentionMaskTensors)
        .fetch(_tfBartSignatures
          .getOrElse(ModelSignatureConstants.DecoderOutput.key, "missing_output_0"))

      val decoderOuts = runner.run().asScala
      val decoderOutputs = TensorResources
        .extractFloats(decoderOuts.head)
        .grouped(vocab_size)
        .toArray
        .grouped(decoderInputLength)
        .toArray
      var nextTokenLogits = for (decoderOutput <- decoderOutputs) yield decoderOutput.last

      nextTokenLogits = nextTokenLogits.map(logits => {
        logits.indices
          .map(i => {
            if (ignoreTokenIds.contains(i)) Float.MinValue else logits(i)
          })
          .toArray
      })

      // repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
      if (repetitionPenalty != 1.0) {
        nextTokenLogits =
          createNextTokenLogitsPenalties(decoderInputs, nextTokenLogits, repetitionPenalty)
      }

      if (noRepeatNgramSize > 0) {
        // calculate a list of banned tokens to prevent repetitively generating the same ngrams
        // from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
        val bannedTokens =
          calcBannedNgramTokens(decoderInputs, batch_size, noRepeatNgramSize, curLen)
        // create bannedTokens boolean mask
        var bannedTokensIndicesMask = Array.empty[IndexedSeq[Boolean]]
        for (bannedTokensSlice <- bannedTokens) {
          bannedTokensIndicesMask = bannedTokensIndicesMask :+
            (for (token <- 0 until vocab_size)
              yield if (bannedTokensSlice.contains(token)) true else false)
        }
        if (!bannedTokensIndicesMask.isEmpty)
          nextTokenLogits =
            for ((nextTokenLogit, bannedTokensIndexMask) <- nextTokenLogits.zip(
                bannedTokensIndicesMask))
              yield setTensorByIndicesToValue(
                nextTokenLogit,
                bannedTokensIndexMask,
                Float.NegativeInfinity)
      }

      // set eos token prob to zero if minLength is not reached
      if (!eosTokenId.isNaN && curLen < minOutputLength) {
        // create eosTokenId boolean mask
        val isTokenLogit_eosToken =
          for (token <- 0 until vocab_size)
            yield if (token == eosTokenId) true else false

        val eosTokenIndices_mask = Array.fill(batch_size)(isTokenLogit_eosToken)

        nextTokenLogits =
          for ((nextTokenLogit, bannedTokensIndex_mask) <- nextTokenLogits.zip(
              eosTokenIndices_mask))
            yield setTensorByIndicesToValue(
              nextTokenLogit,
              bannedTokensIndex_mask,
              Float.NegativeInfinity)
      }

      var nextToken = Array.ofDim[Long](decoderInputs.length)

      if (doSample) {
        // Temperature (higher temperature => more likely to sample low probability tokens)
        if (temperature != 1.0)
          nextTokenLogits =
            for (nextTokenLogit <- nextTokenLogits)
              yield nextTokenLogit.map(_ / temperature.toFloat)
        // Top-p/top-k filtering
        nextTokenLogits = topKTopPFiltering(nextTokenLogits, topK, topP)
        // Sample
        nextToken = nextTokenLogits.map(input => categoricalSample(input, randomSeed))
      } else {
        // Greedy decoding
        nextToken = nextTokenLogits.map(input => input.indexOf(input.max).toLong)
      }
      var tokensToAdd = Array.ofDim[Long](decoderInputs.length)

      // update generations and finished sentences
      if (!eosTokenId.isNaN)
        // pad finished sentences if eos_token_id exist
        tokensToAdd =
          nextToken.zip(unfinishedSents).map(x => x._1 * x._2 + paddingTokenId * (1 - x._2))
      else
        tokensToAdd = nextToken

      decoderInputs = decoderInputs
        .zip(tokensToAdd)
        .map(x => {
          x._1 ++ Array(x._2)
        })
      decoderOuts.foreach(_.close())

      curLen += 1

      if (!eosTokenId.isNaN) {
        val eosInSents = tokensToAdd.zipWithIndex.map { case (x, ind) =>
          if (x == eosTokenId && ind != 0) 1 else 0
        }
        // if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
        val isSentsUnfinishedAndTokenToAddIsEos =
          unfinishedSents.zip(eosInSents).map(x => x._1 * x._2)

        sentLengths = sentLengths
          .zip(isSentsUnfinishedAndTokenToAddIsEos)
          .map(x => x._1 * (1 - x._2) + curLen * x._2)

        // unfinishedSents is set to zero if eos in sentence
        unfinishedSents =
          unfinishedSents.zip(isSentsUnfinishedAndTokenToAddIsEos).map(x => x._1 - x._2)
      }

      tensorDecoder.clearTensors()
      tensorDecoder.clearSession(decoderOuts)
      decoderInputTensors.close()

      // stop when there is a eos in each sentence, or if we exceed the maximum length
      //      stopDecoder = curLen < maxOutputLength || unfinishedSents.max == 0
      stopDecoder = (!decoderInputs.exists(o => !(o.count(_ == this.eosTokenId) == 2))
        || (decoderInputs.head.length > maxOutputLength))

    }
//    tensorEncoder.clearTensors()
    decoderInputs
  }

  def createNextTokenLogitsPenalties(
      inputIds: Seq[Array[Long]],
      logits: Array[Array[Float]],
      repetitionPenalty: Double): Array[Array[Float]] = {
    // create logit penalties for already seen inputIds
    val nextTokenLogits = Array.ofDim[Array[Float]](logits.length)

    for (i <- logits.indices) {
      var nextTokenLogit = logits(i)
      val prevInputIds = inputIds.head.distinct
      for ((prevInputId, _) <- prevInputIds.zipWithIndex) {
        var logitPenalty = 1.0
        if (logits(i)(prevInputId.toInt) < 0) {
          logitPenalty = repetitionPenalty
        } else {
          logitPenalty = 1 / repetitionPenalty
        }
        nextTokenLogit = nextTokenLogit.updated(
          prevInputId.toInt,
          (logitPenalty * nextTokenLogit(prevInputId.toInt)).toFloat)
      }
      nextTokenLogits(i) = nextTokenLogit
    }
    nextTokenLogits
  }

  private def calcBannedNgramTokens(
      prevInputIds: Seq[Array[Long]],
      numHypos: Int,
      noRepeatNgramSize: Int,
      curLen: Int): Array[Array[Long]] = {
    // based on fairseq for noRepeatNgram in beam_search
    if (curLen + 1 < noRepeatNgramSize)
      // return no banned tokens if we haven't generated noRepeatNgram_size tokens yet
      return Array.ofDim[Long](numHypos, 0)
    val generatedNgrams =
      Array.tabulate(numHypos)(_ => mutable.Map.empty[IndexedSeq[Long], List[Long]])
    for (idx <- 0 until numHypos) {
      val genTokens = prevInputIds(idx)
      val generatedNgram = generatedNgrams(idx)
      val ngramArrays = for (e <- 0 until noRepeatNgramSize) yield genTokens.drop(e)
      for (ngramInd <- ngramArrays.last.indices) {
        val ngram = for (e <- ngramArrays) yield e(ngramInd)
        val prevNgramTuple = ngram.dropRight(1)
        generatedNgram(prevNgramTuple) =
          generatedNgram.getOrElse(prevNgramTuple, List.empty[Long]) :+ ngram.last
      }
    }
    (for (hypoIdx <- 0 until numHypos)
      yield getGeneratedNgrams(
        prevInputIds,
        generatedNgrams,
        hypoIdx,
        curLen,
        noRepeatNgramSize)).toArray
  }

  def getGeneratedNgrams(
      prevInputIds: Seq[Array[Long]],
      generatedNgrams: Array[mutable.Map[IndexedSeq[Long], List[Long]]],
      hypoIdx: Int,
      curLen: Int,
      noRepeatNgramSize: Int): Array[Long] = {
    // Before decoding the next token, prevent decoding of ngrams that have already appeared
    val startIdx = curLen + 1 - noRepeatNgramSize
    val ngramIdx = prevInputIds(hypoIdx).slice(startIdx, curLen)
    generatedNgrams(hypoIdx).getOrElse(ngramIdx, List.empty[Long]).toArray
  }

  private def topKTopPFiltering(
      logits: Array[Array[Float]],
      topK: Int,
      topP: Double,
      filterValue: Float = Float.NegativeInfinity,
      minTokensToKeep: Int = 1): Array[Array[Float]] = {

    /** Filter a distribution of logits using top-k and/or nucleus (top-p) filtering * Args:
      * logits: logits distribution shape (batch size, vocabulary size) if topK > 0: keep only top
      * k tokens with highest probability (top-k filtering). if topP < 1.0: keep the top tokens
      * with cumulative probability >= topP (nucleus filtering). Nucleus filtering is described in
      * Holtzman et al. (http://arxiv.org/abs/1904.09751) Make sure we keep at least
      * minTokensToKeep per batch example in the output From:
      * https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
      */
    var logitsUpd = logits
    val logitsShape = Array(logits.length, logits(0).length)

    if (topK > 0) {
      val topKup = topK.max(minTokensToKeep).min(logitsShape.last) // Safety check

      /** Remove all tokens with a probability less than the last token of the top-k */
      val removeLimit = logits(0).sortWith(_ > _).take(topKup).min
      val indicesToRemove =
        for (logit <- logits)
          yield for (elem <- logit) yield if (elem < removeLimit) true else false

      logitsUpd =
        for ((nextTokenLogit, indexToRemove) <- logits.zip(indicesToRemove))
          yield setTensorByIndicesToValue(nextTokenLogit, indexToRemove, Float.NegativeInfinity)
    }
    if (topP < 1.0) {
      val (sortedLogits, sortedIndices) = logits(0).zipWithIndex.sorted.reverse.unzip

      val cumulativeProbs = scanLeft(softmax(sortedLogits))(0.0)(_ + _).drop(1)

      /** Remove tokens with cumulative probability above the threshold (token with 0 are kept) */
      var sortedIndicesToRemove =
        for (prob <- cumulativeProbs)
          yield if (prob > topP) true else false

      if (minTokensToKeep > 1) {

        /** Keep at least minTokensToKeep (set to minTokensToKeep-1 because we add the first one
          * below)
          */
        sortedIndicesToRemove = List.fill(sortedIndicesToRemove.take(minTokensToKeep).length)(
          false) ++ sortedIndicesToRemove.drop(minTokensToKeep)
      }

      /** Shift the indices to the right to keep also the first token above the threshold */
      sortedIndicesToRemove = sortedIndicesToRemove.takeRight(1) ++ sortedIndicesToRemove
        .dropRight(1)
      sortedIndicesToRemove =
        List.fill(sortedIndicesToRemove.take(1).length)(false) ++ sortedIndicesToRemove
          .drop(1)

      /** scatter sorted tensors to original indexing */
      val indicesToRemove = scatterValuesOnBatchIndices(sortedIndicesToRemove, sortedIndices)
      logitsUpd =
        for ((nextTokenLogit, indexToRemove) <- logits.zip(
            IndexedSeq.fill(logits.length)(indicesToRemove)))
          yield setTensorByIndicesToValue(
            nextTokenLogit,
            indexToRemove.toIndexedSeq,
            Float.NegativeInfinity)
    }
    logitsUpd
  }

  private def scanLeft[a, b](xs: Iterable[a])(s: b)(f: (b, a) => b) =
    xs.foldLeft(List(s))((acc, x) => f(acc.head, x) :: acc).reverse

  private def scatterValuesOnBatchIndices(
      values: List[Boolean],
      batchIndices: Array[Int]): List[Boolean] = {
    // scatter values to pair indices
    val (_, initArray) = batchIndices.zip(values).sorted.unzip
    initArray.toList
  }

//  private def softmax(values: Array[Float]): Array[Float] = {
//    val expElem = values.map(exp(_))
//    val total = expElem.sum
//    expElem.map(_ / total).map(_.toFloat)
//  }

  private def setTensorByIndicesToValue(
      prevInputIds: Array[Float],
      indices: IndexedSeq[Boolean],
      value: Float): Array[Float] = {
    for ((inputId, index) <- prevInputIds.zip(indices)) yield if (index) value else inputId
  }

  private def categoricalSample(dist: Array[Float], randomSeed: Option[Long]): Long = {
    val (distFiltered, indices) =
      dist.zipWithIndex.filter { case (elem, index) => !elem.isInfinite }.sorted.unzip

    if (distFiltered.length == 1)
      return indices(0)

    //    val distMinValue = distFiltered.min
    //    val distRange = distFiltered.max - distMinValue
    //    val normalized = distFiltered.map(i => (i - distMinValue)/distRange)
    val normalized = softmax(distFiltered)

    var randomDouble = 0.0
    if (randomSeed.isDefined)
      randomDouble = new scala.util.Random(randomSeed.get).nextDouble()
    else
      randomDouble = scala.util.Random.nextDouble()

    var accum = 0.0
    for ((itemProb, i) <- normalized.zip(indices)) {
      accum += itemProb
      if (accum >= randomDouble) {
        return i
      }
    }
    indices(0)
  }

  def decode(sentences: Array[Array[Long]]): Seq[String] = {
    sentences.map(s => bpeTokenizer.decodeTokens(s.map(_.toInt)))
  }

  def encode(sentences: Seq[Annotation], task: String): Seq[Array[Long]] = {
    SentenceSplit
      .unpack(sentences)
      .map(s => {
        val sentWithTask =
          if (task.nonEmpty) s
//            new Sentence(
//              content = task.concat(" ").concat(s.content),
//              start = s.start,
//              end = s.end + task.length + 1,
//              index = s.index,
//              metadata = s.metadata)
          else s
        bpeTokenizer
          .tokenize(sentWithTask)
          .map(bpeTokenizer.encode)
          .flatMap(_.map(_.pieceId.toLong))
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
      ignoreTokenIds: Array[Long] = Array(),
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
      encoderInputIds: Seq[Array[Long]],
      decoderInputIds: Seq[Array[Long]],
      maxLength: Int): Array[Array[Float]] = {
    val sequencesLength = encoderInputIds.map(x => x.length).toArray
    var maxSentenceLength = sequencesLength.max // - curLen
    maxSentenceLength = Math.max(maxSentenceLength, maxLength)
    val vocab_size = this.vocab_size

    val decoderInputLength = decoderInputIds.head.length
    val tensorDecoder = new TensorResources()

    val decoderInputBuffers =
      tensorDecoder.createLongBuffer(decoderInputIds.length * decoderInputLength)
    val decoderAttentionBuffers =
      tensorDecoder.createLongBuffer(decoderInputIds.length * decoderInputLength)

    decoderInputIds.zipWithIndex.foreach { case (pieceIds, idx) =>
      val offset = idx * decoderInputLength
      decoderInputBuffers.offset(offset).write(pieceIds)
      val paddingMasks = pieceIds.map(_ => 1L)
      decoderAttentionBuffers.offset(offset).write(paddingMasks)
    }

    val decoderInputTensors = tensorDecoder.createLongBufferTensor(
      Array(decoderInputIds.length.toLong, decoderInputLength),
      decoderInputBuffers)
    val decoderAttentionMaskTensors = tensorDecoder.createLongBufferTensor(
      Array(decoderInputIds.length.toLong, decoderInputLength),
      decoderAttentionBuffers)

    val runner = this.session.runner

    // TODO add past to the model and use cache
    runner
      .feed(
        _tfBartSignatures.getOrElse(
          ModelSignatureConstants.DecoderInputIds.key,
          "missing_decoder_input_ids"),
        decoderInputTensors)
      .feed(
        _tfBartSignatures.getOrElse(
          ModelSignatureConstants.DecoderAttentionMask.key,
          "missing_encoder_attention_mask"),
        decoderAttentionMaskTensors)
      .feed(
        _tfBartSignatures.getOrElse(
          ModelSignatureConstants.DecoderEncoderInputIds.key,
          "missing_encoder_state"),
        decoderEncoderStateTensors)
      .feed(
        _tfBartSignatures.getOrElse(
          ModelSignatureConstants.DecoderEncoderAttentionMask.key,
          "missing_decoder_encoder_attention_mask"),
        encoderAttentionMaskTensors)
      .fetch(_tfBartSignatures
        .getOrElse(ModelSignatureConstants.DecoderOutput.key, "missing_output_0"))

    val decoderOuts = runner.run().asScala
    val decoderOutputs = TensorResources
      .extractFloats(decoderOuts.head)
      .grouped(vocab_size)
      .toArray
      .grouped(decoderInputLength)
      .toArray
    var nextTokenLogits = for (decoderOutput <- decoderOutputs) yield decoderOutput.last

//    val nextTokenLogits = for (decoderOutput <- decoderOutputs) yield decoderOutput.last
//    val nextTokenLogits = lastOut.last
//      for (tokens <- decoderOutput) yield tokens.last
//    }
    tensorDecoder.clearTensors()
    tensorDecoder.clearSession(decoderOuts)
    decoderInputTensors.close()
    nextTokenLogits
  }
}
