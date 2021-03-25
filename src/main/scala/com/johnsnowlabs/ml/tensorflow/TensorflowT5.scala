package com.johnsnowlabs.ml.tensorflow

import com.johnsnowlabs.ml.tensorflow.sentencepiece._
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import org.tensorflow.{Session, Tensor}

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.math._

/**
  * This class is used to run T5 model for For Sequence Batches of WordpieceTokenizedSentence.
  * Input for this model must be tokenized with a SentencePieceModel,
  *
  * @param tensorflow       Albert Model wrapper with TensorFlowWrapper
  * @param spp              Albert SentencePiece model with SentencePieceWrapper
  * @param configProtoBytes Configuration for TensorFlow session
  */

class TensorflowT5(val tensorflow: TensorflowWrapper,
                   val spp: SentencePieceWrapper,
                   configProtoBytes: Option[Array[Byte]] = None
                  ) extends Serializable {

  // keys representing the input and output tensors of the T5 model

  private val encoderInputIdsKey = "encoder_input_ids:0"
  private val encoderAttentionMaskKey = "encoder_attention_mask:0"
  private val decoderInputIdsKey = "decoder_input_ids:0"
  private val decoderEncoderStateKey = "encoder_state:0"
  //  private val decoderPastKey = "decoderPastKeyValues:0"
  //  private val decoderCacheKey = "decoder_use_cache:0"
  private val decoderEncoderAttentionMaskKey = "decoder_encoder_attention_mask:0"
  private val decoderAttentionMaskKey = "decoder_attention_mask:0"

  private val encoderOutputsKey = "encoder_outputs:0"
  private val decoderOutputsKey = "decoder_outputs:0"

  private val paddingTokenId = 0L
  private val eosTokenId = 1L
  private val pieceSize = spp.getSppModel.getPieceSize

  def generateSeq2Seq(sentences: Seq[Annotation],
                      batchSize: Int = 1,
                      maxOutputLength: Int,
                      task: String
                     ): Seq[Annotation] = {

    val batchDecoder = sentences.grouped(batchSize).toArray.flatMap { batch =>

      val batchSP = encode(batch, task)
      val spIds = process(batchSP, maxOutputLength)
      decode(spIds)

    }

    var sentBegin, nextSentEnd = 0
    batchDecoder.zip(sentences).map {
      case (content, sent) =>
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

  def process(batch: Seq[Array[Long]], minOutputLength: Int = 10, maxOutputLength: Int = 20, doSample: Boolean = false, temperature: Double = 1.0, topK: Int = 50, topP: Double = 1.0, repetitionPenalty: Double = 1.0, noRepeatNgramSize: Int = 0): Array[Array[Long]] = {

    /* Actual size of each sentence to skip padding in the TF model */
    val sequencesLength = batch.map(x => x.length).toArray
    val maxSentenceLength = sequencesLength.max // - curLen

    val num_beams = 1
    val num_return_sequences = 1
    //from config
    val vocab_size = 32128

    //////////////////////
    var effective_batch_size = 1
    var effective_batch_mult = 1

    //    val inputDim = batch.length * maxSentenceLength
    // set effective batch size and effective batch multiplier according to do_sample
    if (doSample) {
      effective_batch_size = batch.length * num_return_sequences
      effective_batch_mult = num_return_sequences
    }
    else {
      effective_batch_size = batch.length
      effective_batch_mult = 1
    }

    //Run encoder
    val encoderInputTensorResources = new TensorResources()
    val encoderAttentionMaskTensorResources = new TensorResources()

    val inputDim = batch.length * maxSentenceLength

    val encoderInputBuffers = encoderInputTensorResources.createLongBuffer(inputDim)
    val encoderAttentionMaskBuffers = encoderAttentionMaskTensorResources.createLongBuffer(inputDim)

    val shape = Array(batch.length.toLong, maxSentenceLength)

    batch.foreach(tokenIds => {

      val diff = maxSentenceLength - tokenIds.length

      val s = tokenIds.take(maxSentenceLength) ++ Array.fill[Long](diff)(this.paddingTokenId)
      encoderInputBuffers.put(s)
      val mask = s.map(x => if (x != this.paddingTokenId) 1L else 0L)
      encoderAttentionMaskBuffers.put(mask)
    })

    encoderInputBuffers.flip()
    encoderAttentionMaskBuffers.flip()

    val encoderInputTensors = encoderInputTensorResources.createLongBufferTensor(shape, encoderInputBuffers)
    val encoderAttentionMaskTensors = encoderAttentionMaskTensorResources.createLongBufferTensor(shape, encoderAttentionMaskBuffers)

    val session = tensorflow.getTFHubSession(configProtoBytes = configProtoBytes)
    val runner = session.runner

    runner
      .feed(encoderInputIdsKey, encoderInputTensors)
      .feed(encoderAttentionMaskKey, encoderAttentionMaskTensors)
      .fetch(encoderOutputsKey)

    val encoderOuts = runner.run().asScala
    val encoderOutsFloats = TensorResources.extractFloats(encoderOuts.head)
    val dim = encoderOutsFloats.length / inputDim
    val encoderOutsBatch = encoderOutsFloats.grouped(dim).toArray.grouped(maxSentenceLength).toArray

    encoderInputBuffers.clear()

    encoderInputTensorResources.clearTensors()
    encoderInputTensorResources.clearSession(encoderOuts)

    //Run decoder
    val decoderEncoderStateTensorResources = new TensorResources()
    val decoderEncoderStateBuffers = decoderEncoderStateTensorResources.createFloatBuffer(batch.length * maxSentenceLength * dim)
    batch.zipWithIndex.foreach(bi => {
      encoderOutsBatch(bi._2).foreach(encoderOutput => {
        decoderEncoderStateBuffers.put(encoderOutput)
      })
    })
    decoderEncoderStateBuffers.flip()
    val decoderEncoderStateTensors = encoderInputTensorResources.createFloatBufferTensor(
      Array(batch.length.toLong, maxSentenceLength, dim),
      decoderEncoderStateBuffers)

    val modelOutputs = _generateNo_beam_search(batch, decoderEncoderStateTensors, encoderAttentionMaskTensors, maxOutputLength, minOutputLength, doSample,
      temperature, topK, topP, repetitionPenalty, noRepeatNgramSize, effective_batch_size, vocab_size, session)

    encoderAttentionMaskBuffers.clear()
    encoderAttentionMaskTensorResources.clearTensors()

    decoderEncoderStateBuffers.clear()
    decoderEncoderStateTensorResources.clearTensors()

    modelOutputs
  }

  def _generateNo_beam_search(inputIds: Seq[Array[Long]],
                              decoderEncoderStateTensors: Tensor[_],
                              encoderAttentionMaskTensors: Tensor[_],
                              maxOutputLength: Int,
                              minOutputLength: Int,
                              doSample: Boolean,
                              temperature: Double,
                              topK: Int,
                              topP: Double,
                              repetitionPenalty: Double,
                              noRepeatNgramSize: Int,
                              batch_size: Int,
                              vocab_size: Int,
                              session: Session): Array[Array[Long]] = {

    /**
      * Generate sequences for each example without beam search (num_beams == 1). All returned sequence are generated
      * independently.
      **/

    var decoderInputs = inputIds.map(_ => Array(this.paddingTokenId)).toArray
    var modelOutputs = inputIds.map(_ => Array(this.paddingTokenId)).toArray

    var curLen = modelOutputs(0).length

    var stopDecoder = false
    var past = decoderEncoderStateTensors

    while (!stopDecoder) {
      val decoderInputLength = decoderInputs.head.length
      val decoderInputTensorResources = new TensorResources()
      val decoderAttentionTensorResources = new TensorResources()
      val decoderInputBuffers = decoderInputTensorResources.createLongBuffer(inputIds.length * decoderInputLength)
      val decoderAttentionBuffers = decoderAttentionTensorResources.createLongBuffer(inputIds.length * decoderInputLength)

      inputIds.zipWithIndex.foreach(bi => {
        decoderInputs(bi._2).zipWithIndex.foreach(x => {
          decoderInputBuffers.put(x._1)
          decoderAttentionBuffers.put(if ((x._2 != 0) && (x._1 == this.paddingTokenId)) 0L else 1L)
        })
      })

      decoderInputBuffers.flip()
      decoderAttentionBuffers.flip()

      // decoderInputIds
      val decoderInputTensors = decoderInputTensorResources.createLongBufferTensor(
        Array(inputIds.length.toLong, decoderInputLength), decoderInputBuffers)
      val decoderAttentionMaskTensors = decoderAttentionTensorResources.createLongBufferTensor(
        Array(inputIds.length.toLong, decoderInputLength), decoderAttentionBuffers)
      val runner = session.runner

      // TODO add past to the model and use cache
      runner
        .feed(decoderInputIdsKey, decoderInputTensors)
        .feed(decoderEncoderStateKey, decoderEncoderStateTensors)
        .feed(decoderEncoderAttentionMaskKey, encoderAttentionMaskTensors)
        .feed(decoderAttentionMaskKey, decoderAttentionMaskTensors)
        .fetch(decoderOutputsKey)

      val decoderOuts = runner.run().asScala
      var decoderOutputs = TensorResources.extractFloats(decoderOuts.head).grouped(32128).toArray.grouped(decoderInputLength).toArray
      var nextTokenLogits = for (decoderOutput <- decoderOutputs) yield decoderOutput.last

      // repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
      if (repetitionPenalty != 1.0) {
        nextTokenLogits = createNextTokenLogitsPenalties(
          modelOutputs, nextTokenLogits, repetitionPenalty
        )
      }

      if (noRepeatNgramSize > 0) {
        // calculate a list of banned tokens to prevent repetitively generating the same ngrams
        // from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
        val bannedTokens = calc_bannedNgramTokens(modelOutputs, batch_size, noRepeatNgramSize, curLen)
        // create bannedTokens boolean mask
        var bannedTokensIndicesMask = Array.empty[IndexedSeq[Boolean]]
        for (bannedTokensSlice <- bannedTokens) {
          if (!bannedTokensSlice.isEmpty)
            bannedTokensIndicesMask = bannedTokensIndicesMask :+
              (for (token <- (0 to vocab_size - 1)) yield if (bannedTokensSlice.contains(token)) true else false)
        }
        if (!bannedTokensIndicesMask.isEmpty)
          nextTokenLogits = for ((nextTokenLogit, bannedTokensIndexMask) <- nextTokenLogits.zip(bannedTokensIndicesMask)) yield setTensorByIndicesToValue(
            nextTokenLogit, bannedTokensIndexMask, Float.NegativeInfinity
          )
      }

      // set eos token prob to zero if minLength is not reached
      if (!eosTokenId.isNaN && curLen < minOutputLength) {
        // create eosTokenId boolean mask
        val isTokenLogit_eosToken = for (token <- (0 to vocab_size - 1)) yield if (token == eosTokenId) true else false

        val eosTokenIndices_mask = Array.fill(batch_size)(isTokenLogit_eosToken)

        nextTokenLogits = for ((nextTokenLogit, bannedTokensIndex_mask) <- nextTokenLogits.zip(eosTokenIndices_mask)) yield setTensorByIndicesToValue(
          nextTokenLogit, bannedTokensIndex_mask, Float.NegativeInfinity
        )
      }

      var nextToken = 0L

      if (doSample) {
        // Temperature (higher temperature => more likely to sample low probability tokens)
        if (temperature != 1.0)
          nextTokenLogits = for (nextTokenLogit <- nextTokenLogits) yield nextTokenLogit.map(_ / temperature.toFloat)
        // Top-p/top-k filtering
        nextTokenLogits = topKTopPFiltering(nextTokenLogits, topK, topP)
        // Sample
        nextToken = nextTokenLogits.map(input => categoricalSample(input)).last.toLong
      }
      else {
        // Greedy decoding
        nextToken = nextTokenLogits.map(input => input.indexOf(input.max)).last.toLong
      }
      val tt = Array.fill(batch_size)(nextToken)
      //      val outputIds = decoderOutputs.map(batch => batch.map(input => input.indexOf(input.max)).last).map(_.toLong)
      decoderInputs = decoderInputs.zip(Array.fill(batch_size)(nextToken)).map(x => x._1 ++ Array(x._2))
      modelOutputs = modelOutputs.zip(Array.fill(batch_size)(nextToken)).map(x => {
        if (x._1.contains(eosTokenId)) {
          x._1
        } else {
          x._1 ++ Array(x._2)
        }
      })
      curLen += 1

      decoderOuts.foreach(_.close())

      decoderInputBuffers.clear()
      decoderInputTensorResources.clearTensors()
      decoderAttentionBuffers.clear()

      stopDecoder = (
        !modelOutputs.exists(o => o.last != this.eosTokenId)
          || (modelOutputs.head.length > maxOutputLength))

    }
    modelOutputs
  }

  def createNextTokenLogitsPenalties(inputIds: Seq[Array[Long]], logits: Array[Array[Float]], repetitionPenalty: Double): Array[Array[Float]] = {
    // create logit penalties for already seen inputIds
    var nextTokenLogits = Array.ofDim[Array[Float]](logits.size)

    for (i <- logits.indices) {
      var nextTokenLogit = logits(i)
      val prevInputIds = inputIds(0).distinct
      for ((prevInputId, j) <- prevInputIds.zipWithIndex) {
        var logitPenalty = 1.0
        if (logits(i)(prevInputId.toInt) < 0) {
          logitPenalty = repetitionPenalty
        }
        else {
          logitPenalty = 1 / repetitionPenalty
        }
        nextTokenLogit = nextTokenLogit.updated(prevInputId.toInt, (logitPenalty * nextTokenLogit(prevInputId.toInt)).toFloat)
      }
      nextTokenLogits(i) = nextTokenLogit
    }
    nextTokenLogits
  }

  private def calc_bannedNgramTokens(prevInputIds: Seq[Array[Long]], numHypos: Int, noRepeatNgramSize: Int, curLen: Int): Array[Array[Long]] = {
    // based on fairseq for no_repeatNgram in beam_search
    if (curLen + 1 < noRepeatNgramSize)
    // return no banned tokens if we haven't generated no_repeatNgram_size tokens yet
      return Array.ofDim[Long](numHypos, 0)
    var generatedNgrams = Array.tabulate(numHypos)(_ => mutable.Map.empty[IndexedSeq[Long], List[Long]])
    for (idx <- 0 to numHypos - 1) {
      val genTokens = prevInputIds(idx)
      var generatedNgram = generatedNgrams(idx)
      val ngramArrays = for (e <- 0 to noRepeatNgramSize - 1) yield genTokens.drop(e)
      for (ngramInd <- 0 to (ngramArrays.last.length - 1)) {
        val ngram = for (e <- ngramArrays) yield e(ngramInd)
        val prevNgramTuple = ngram.dropRight(1)
        generatedNgram(prevNgramTuple) = generatedNgram.getOrElse(prevNgramTuple, List.empty[Long]) :+ ngram.last
      }
    }
    return (for (hypoIdx <- 0 to numHypos - 1) yield getGeneratedNgrams(prevInputIds, generatedNgrams, hypoIdx, curLen, noRepeatNgramSize)).toArray
  }

  def getGeneratedNgrams(prevInputIds: Seq[Array[Long]], generatedNgrams: Array[mutable.Map[IndexedSeq[Long], List[Long]]], hypoIdx: Int, curLen: Int, noRepeatNgramSize: Int): Array[Long] = {
    // Before decoding the next token, prevent decoding of ngrams that have already appeared
    val startIdx = curLen + 1 - noRepeatNgramSize
    val ngramIdx = prevInputIds(hypoIdx).slice(startIdx, curLen)
    generatedNgrams(hypoIdx).getOrElse(ngramIdx, List.empty[Long]).toArray
  }

  private def topKTopPFiltering(logits: Array[Array[Float]], topK: Int, topP: Double, filterValue: Float = Float.NegativeInfinity, minTokensToKeep: Int = 1): Array[Array[Float]] = {
    /**
      * Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
      * *
      * Args:
      * logits: logits distribution shape (batch size, vocabulary size)
      * if topK > 0: keep only top k tokens with highest probability (top-k filtering).
      * if topP < 1.0: keep the top tokens with cumulative probability >= topP (nucleus filtering).
      * Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
      * Make sure we keep at least minTokensToKeep per batch example in the output
      * From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
      **/

    var logitsUpd = logits
    val logitsShape = Array(logits.length, logits(0).length)

    if (topK > 0) {
      val topKup = topK.max(minTokensToKeep).min(logitsShape.last) // Safety check
      // Remove all tokens with a probability less than the last token of the top-k
      val removeLimit = logits(0).sortWith(_ > _).take(topKup).min
      val indicesToRemove = for (logit <- logits) yield for (elem <- logit) yield if (elem < removeLimit) true else false

      logitsUpd = for ((nextTokenLogit, indexToRemove) <- logits.zip(indicesToRemove)) yield setTensorByIndicesToValue(
        nextTokenLogit, indexToRemove, Float.NegativeInfinity
      )
    }

    if (topP < 1.0) {
      val sortedIndices = logits(0).zipWithIndex.sorted.unzip._2.reverse
      val sortedLogits = logits(0).sortWith(_ > _) // expects logits to be of dim (batch_size, vocab_size)

      val cumulativeProbs = scanLeft(softmax(sortedLogits))(0.0)(_ + _).drop(1)
      //    Remove tokens with cumulative probability above the threshold (token with 0 are kept)
      var sortedIndicesToRemove = for (prob <- cumulativeProbs) yield if (prob > topP) true else false

      if (minTokensToKeep > 1) {
        //    Keep at least minTokensToKeep (set to minTokensToKeep-1 because we add the first one below)
        sortedIndicesToRemove = List.fill(sortedIndicesToRemove.take(minTokensToKeep).length)(false) ++ sortedIndicesToRemove.drop(minTokensToKeep)
      }
      //    Shift the indices to the right to keep also the first token above the threshold
      sortedIndicesToRemove = sortedIndicesToRemove.takeRight(1) ++ sortedIndicesToRemove.dropRight(1)
      sortedIndicesToRemove = List.fill(sortedIndicesToRemove.take(1).length)(false) ++ sortedIndicesToRemove.drop(1)
      //    scatter sorted tensors to original indexing
      val indicesToRemove = scatterValuesOnBatchIndices(sortedIndicesToRemove, sortedIndices)
      logitsUpd = for ((nextTokenLogit, indexToRemove) <- logits.zip(IndexedSeq.fill(logits.length)(indicesToRemove))) yield setTensorByIndicesToValue(
        nextTokenLogit, indexToRemove.toIndexedSeq, Float.NegativeInfinity
      )
    }
    return logitsUpd
  }

  private def scanLeft[a, b](xs: Iterable[a])(s: b)(f: (b, a) => b) =
    xs.foldLeft(List(s))((acc, x) => f(acc(0), x) :: acc).reverse

  private def scatterValuesOnBatchIndices(values: List[Boolean], batchIndices: Array[Int]): List[Boolean] = {
    // scatter values to pair indices
    var initArray = List.fill(batchIndices.length)(false)
    for ((batchIndex, i) <- batchIndices.zipWithIndex) {
      initArray = initArray.updated(batchIndex, values(i))
    }
    return initArray
  }

  private def softmax(values: Array[Float]): Array[Float] = {
    val expElem = values.map(exp(_))
    val total = expElem.sum
    expElem.map(_ / total).map(_.toFloat)
  }

  private def setTensorByIndicesToValue(prevInputIds: Array[Float], indices: IndexedSeq[Boolean], value: Float): Array[Float] = {
    for ((inputId, index) <- prevInputIds.zip(indices)) yield if (index) value else inputId
  }

  private def categoricalSample(dist: Array[Float]): Int = {
    val (distFiltered, indices) = dist.zipWithIndex.filter { case (elem, index) => !elem.isInfinite }.sorted.unzip

    if (distFiltered.length == 1)
      return indices(0)
    var normalized = (distFiltered.map(_ - distFiltered.min)).map(_ / (distFiltered.max - distFiltered.min))

    val p = scala.util.Random.nextDouble()
    var accum = 0.0
    for ((itemProb, i) <- normalized.zip(indices)) {
      accum += itemProb
      if (accum >= p)
        return i
    }
    return indices(0)
  }

  def decode(sentences: Array[Array[Long]]): Seq[String] = {

    sentences.map { s =>
      val filteredPieceIds = s.filter(x => x <= pieceSize)
      spp.getSppModel.decodeIds(filteredPieceIds.map(_.toInt): _*)
    }

  }

  def encode(sentences: Seq[Annotation], task: String): Seq[Array[Long]] = {
    sentences.map(
      s => {
        val sentWithTask = if (task.nonEmpty) task.concat(" ").concat(s.result) else s.result
        spp.getSppModel.encodeAsIds(sentWithTask).map(_.toLong) ++ Array(this.eosTokenId)
      })
  }

}