package com.johnsnowlabs.ml.ai.seq2seq

import com.johnsnowlabs.ml.openvino.OpenvinoWrapper.EncoderDecoderWrappers
import com.johnsnowlabs.ml.tensorflow.sentencepiece.SentencePieceWrapper
import org.intel.openvino.Tensor

class OpenvinoT5EncoderDecoder(
    val openvinoWrapper: EncoderDecoderWrappers,
    override val spp: SentencePieceWrapper,
    override val additionalTokens: Map[Int, String] = Map())
    extends T5EncoderDecoder(spp, additionalTokens) {

  protected val numLayers: Int = {
    (openvinoWrapper.decoder.getCompiledModel().outputs().size() - 1) / 4
  }

  protected val numAttnHeads: Int = {
    openvinoWrapper.decoderWithPast
      .getCompiledModel()
      .inputs()
      .stream()
      .filter(o => o.get_any_name().equals("past_key_values.0.decoder.value"))
      .findFirst()
      .get()
      .get_partial_shape()
      .get_dimension(1)
      .get_length()
  }

  sessionWarmup()

  override def tag(
      batch: Seq[Array[Int]],
      maxNewTokens: Int,
      maxTextLength: Int,
      doSample: Boolean,
      topK: Int,
      topP: Double,
      temperature: Double,
      noRepeatNgramSize: Int,
      repetitionPenalty: Double,
      randomSeed: Option[Long],
      ignoreTokenIds: Array[Int] = Array(),
      stopAtEos: Boolean): Array[Array[Int]] = {
    val sequencesLength = batch.map(x => x.length).toArray
    val maxSentenceLength = sequencesLength.max

    val numReturn_sequences = 1
    val vocabSize = 32128

    val encInferRequest = openvinoWrapper.encoder.getCompiledModel().create_infer_request()

    // Run encoder
    val encoderInputBuffers = batch
      .flatMap(tokenIds =>
        (tokenIds.take(maxSentenceLength).map(_.toLong) ++ Array.fill[Long](
          maxSentenceLength - tokenIds.length)(this.paddingTokenId)))
      .toArray
    val encoderAttentionMaskBuffers =
      encoderInputBuffers.map(x => if (x != this.paddingTokenId) 1L else 0L)

    val inputShape = Array(batch.length, maxSentenceLength)
    val encoderInputTensors = new Tensor(inputShape, encoderInputBuffers)
    val encoderAttentionMaskTensors = new Tensor(inputShape, encoderAttentionMaskBuffers)

    encInferRequest.set_tensor("input_ids", encoderInputTensors)
    encInferRequest.set_tensor("attention_mask", encoderAttentionMaskTensors)

    encInferRequest.infer()

    val encoderStateTensors = encInferRequest.get_tensor("last_hidden_state")

    val modelOutputs = generateNoBeamSearch(
      batch,
      encoderStateTensors,
      encoderAttentionMaskTensors,
      maxNewTokens = maxNewTokens,
      maxTextLength = maxTextLength,
      doSample = doSample,
      topK = topK,
      topP = topP,
      temperature = temperature,
      vocabSize = vocabSize,
      randomSeed = randomSeed,
      ignoreTokenIds = ignoreTokenIds,
      stopAtEos = stopAtEos,
      noRepeatNgramSize = noRepeatNgramSize,
      repetitionPenalty = repetitionPenalty)

    modelOutputs
  }

  def generateCacheKeys(component: String, state: String): Array[String] = {
    { 0 until numLayers }
      .flatMap(x => Array(s"$state.$x.$component.key", s"$state.$x.$component.value"))
      .toArray
  }

  lazy val encoderCacheInputKeys: Array[String] = generateCacheKeys("encoder", "past_key_values")
  lazy val encoderCacheOutputKeys: Array[String] = generateCacheKeys("encoder", "present")
  lazy val decoderCacheInputKeys: Array[String] = generateCacheKeys("decoder", "past_key_values")
  lazy val decoderCacheOutputKeys: Array[String] = generateCacheKeys("decoder", "present")

  def generateNoBeamSearch(
      inputIds: Seq[Array[Int]],
      encoderStateTensors: Tensor,
      encoderAttentionMaskTensors: Tensor,
      maxNewTokens: Int,
      maxTextLength: Int,
      doSample: Boolean,
      topK: Int,
      topP: Double,
      temperature: Double,
      vocabSize: Int,
      randomSeed: Option[Long],
      ignoreTokenIds: Array[Int] = Array(),
      stopAtEos: Boolean,
      noRepeatNgramSize: Int,
      repetitionPenalty: Double): Array[Array[Int]] = {

    var decoderInputIds = inputIds.map(x => Array(this.paddingTokenId.toLong)).toArray
    val batchSize = decoderInputIds.length
    val decoderInputShape = Array(batchSize, 1)
    val stopTokens = if (stopAtEos) Array(this.eosTokenId) else Array[Int]()
    var decoderInitCache: Option[Array[Tensor]] = None
    var decoderOutputCache: Option[Array[Tensor]] = None

    val decoderProcessor = new DecoderProcessor(
      batchSize = batchSize,
      maxTextLength = maxTextLength,
      sequenceLength = decoderInputIds(0).length,
      doSample = doSample,
      topK = topK,
      topP = topP,
      temperature = temperature,
      vocabSize = vocabSize,
      noRepeatNgramSize = noRepeatNgramSize,
      randomSeed = randomSeed,
      stopTokens = stopTokens,
      ignoreTokenIds = ignoreTokenIds,
      maxNewTokens = maxNewTokens,
      repetitionPenalty = repetitionPenalty,
      paddingTokenId = paddingTokenId)

    while (!decoderProcessor.stopDecoding(decoderInputIds)) {

      var logitsRaw: Array[Float] = Array()
      if (decoderInitCache.isEmpty) {
        // First pass of the decoder
        val decoderInputIdsTensor = new Tensor(decoderInputShape, decoderInputIds.flatten)

        val decoderReq = openvinoWrapper.decoder.getCompiledModel().create_infer_request()
        decoderReq.set_tensor("input_ids", decoderInputIdsTensor)
        decoderReq.set_tensor("encoder_attention_mask", encoderAttentionMaskTensors)
        decoderReq.set_tensor("encoder_hidden_states", encoderStateTensors)

        decoderReq.infer()

        val logitsTensors = decoderReq.get_tensor("logits")
        logitsRaw = logitsTensors.data()
        val decoderInitFetchKeys = encoderCacheOutputKeys ++ decoderCacheOutputKeys
        decoderInitCache = Some(
          decoderInitFetchKeys.map(cacheKey => decoderReq.get_tensor(cacheKey)))
      } else {
        // Subsequent passes of the decoder
        val decoderInputIdsTensor =
          new Tensor(decoderInputShape, decoderInputIds.map(x => x.last))
        val decoderReq = openvinoWrapper.decoderWithPast.getCompiledModel().create_infer_request()

        decoderReq.set_tensor("input_ids", decoderInputIdsTensor)
        decoderReq.set_tensor("encoder_attention_mask", encoderAttentionMaskTensors)
        decoderReq.set_tensor("encoder_hidden_states", encoderStateTensors)

        val decoderFeedKeys = encoderCacheInputKeys.zip(
          decoderInitCache.get.slice(0, encoderCacheInputKeys.length)) ++ (
          if (decoderOutputCache.isEmpty) {
            decoderCacheInputKeys.zip(
              decoderInitCache.get.slice(
                encoderCacheInputKeys.length,
                encoderCacheInputKeys.length + decoderCacheInputKeys.length))
          } else {
            decoderCacheInputKeys.zip(decoderOutputCache.get)
          }
        )
        decoderFeedKeys.foreach { case (k, v) =>
          decoderReq.set_tensor(k, v)
        }

        decoderReq.infer()

        val logitsTensors = decoderReq.get_tensor("logits")
        logitsRaw = logitsTensors.data()

        decoderOutputCache = Some(
          decoderCacheOutputKeys.map(cacheKey => decoderReq.get_tensor(cacheKey)))
      }

      val logits = (0 until batchSize)
        .map(i => {
          logitsRaw.slice(i * vocabSize, (i + 1) * vocabSize)
        })
        .toArray

      decoderInputIds =
        decoderProcessor.processLogits(batchLogits = logits, decoderInputIds = decoderInputIds)

    }
    decoderInputIds.map(x => x.map(_.toInt))
  }
}
