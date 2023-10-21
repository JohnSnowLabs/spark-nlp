package com.johnsnowlabs.ml.ai.seq2seq

import ai.onnxruntime.{OnnxTensor, OrtSession, TensorInfo}
import com.johnsnowlabs.ml.onnx.OnnxWrapper
import com.johnsnowlabs.ml.tensorflow.{TensorResources, TensorflowWrapper}
import com.johnsnowlabs.ml.tensorflow.sentencepiece.SentencePieceWrapper
import com.johnsnowlabs.ml.tensorflow.sign.{ModelSignatureConstants, ModelSignatureManager}

import scala.jdk.CollectionConverters.{mapAsJavaMap, setAsJavaSet}

private[johnsnowlabs] class OnnxMarianEncoderDecoder(
    val onnxEncoder: OnnxWrapper,
    val onnxDecoder: OnnxWrapper,
    override val sppSrc: SentencePieceWrapper,
    override val sppTrg: SentencePieceWrapper)
    extends MarianEncoderDecoder(sppSrc, sppTrg) {

  sessionWarmup()

  protected val numLayers: Int = {
    ((onnxDecoder.getSession()._1.getNumOutputs - 1) / 4).toInt
  }

  protected val numAttnHeads: Int = {
    onnxDecoder
      .getSession()
      ._1
      .getInputInfo
      .get("past_key_values.0.decoder.value")
      .getInfo
      .asInstanceOf[TensorInfo]
      .getShape()(1)
      .toInt
  }

  override protected def tag(
      batch: Seq[Array[Int]],
      maxOutputLength: Int,
      paddingTokenId: Int,
      eosTokenId: Int,
      vocabSize: Int,
      doSample: Boolean = false,
      temperature: Double = 1.0d,
      topK: Int = 50,
      topP: Double = 1.0d,
      repetitionPenalty: Double = 1.0d,
      noRepeatNgramSize: Int = 0,
      randomSeed: Option[Long] = None,
      ignoreTokenIds: Array[Int] = Array()): Array[Array[Int]] = {

    /* Actual size of each sentence to skip padding in the TF model */
    val sequencesLength = batch.map(x => x.length).toArray
    val maxSentenceLength = sequencesLength.max
    val (encoder, env) = onnxEncoder.getSession()

    lazy val encoderCacheInputKeys: Array[String] =
      generateCacheKeys("encoder", "past_key_values")
    lazy val encoderCacheOutputKeys: Array[String] = generateCacheKeys("encoder", "present")
    lazy val decoderCacheInputKeys: Array[String] =
      generateCacheKeys("decoder", "past_key_values")
    lazy val decoderCacheOutputKeys: Array[String] = generateCacheKeys("decoder", "present")

    def generateCacheKeys(component: String, state: String): Array[String] = {
      { 0 until numLayers }
        .flatMap(x => Array(s"$state.$x.$component.key", s"$state.$x.$component.value"))
        .toArray
    }

    val encoderInputBuffers = batch
      .map(tokenIds =>
        (tokenIds.take(maxSentenceLength).map(_.toLong) ++ Array.fill[Long](
          maxSentenceLength - tokenIds.length)(paddingTokenId)))
      .toArray
    val encoderAttentionMaskBuffers =
      encoderInputBuffers.map(x => x.map(xx => if (xx != paddingTokenId) 1L else 0L))

    val encoderInputTensors = OnnxTensor.createTensor(env, encoderInputBuffers)
    val encoderAttentionMaskTensors = OnnxTensor.createTensor(env, encoderAttentionMaskBuffers)

    val encoderResults = encoder.run(
      mapAsJavaMap(
        Map("input_ids" -> encoderInputTensors, "attention_mask" -> encoderAttentionMaskTensors)))

    val encoderStateBuffer =
      try {
        val encoderStateTensor = encoderResults
          .get("last_hidden_state")
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
    encoderInputTensors.close()
    val encoderStateTensors = OnnxTensor.createTensor(env, encoderStateBuffer)

    var decoderInputIds = batch.map(_ => Array(paddingTokenId)).toArray
    val batchSize = decoderInputIds.length

    var decoderInitCache: Option[Array[OnnxTensor]] = None
    var decoderOutputCache: Option[Array[OnnxTensor]] = None
    var decoderInitResults: OrtSession.Result = null
    var decoderResults: OrtSession.Result = null

    val (decoderSession, decoderEnv) = onnxDecoder.getSession()

    val decoderProcessor = new DecoderProcessor(
      batchSize = batchSize,
      maxTextLength = maxOutputLength + maxSentenceLength,
      sequenceLength = decoderInputIds(0).length,
      doSample = doSample,
      topK = topK,
      topP = topP,
      temperature = temperature,
      vocabSize = vocabSize,
      noRepeatNgramSize = noRepeatNgramSize,
      randomSeed = randomSeed,
      stopTokens = Array(eosTokenId),
      paddingTokenId = paddingTokenId,
      ignoreTokenIds = ignoreTokenIds,
      maxNewTokens = maxOutputLength,
      repetitionPenalty = repetitionPenalty)

    while (!decoderProcessor.stopDecoding(decoderInputIds)) {

      val logitsRaw: Array[Float] = if (decoderInitCache.isEmpty) {
        // First pass of the decoder
        val decoderInputIdsTensors =
          OnnxTensor.createTensor(decoderEnv, decoderInputIds.map(x => x.map(_.toLong)))
        // dummy zero tensors for the first pass
        val dummyTensor1 = OnnxTensor.createTensor(
          decoderEnv,
          Array
            .fill(batchSize * batch.head.length * numAttnHeads * 64)(0.0f)
            .grouped(64)
            .toArray
            .grouped(batch.head.length)
            .toArray
            .grouped(numAttnHeads)
            .toArray)
        val dummyTensor2 = OnnxTensor.createTensor(
          decoderEnv,
          Array
            .fill(batchSize * decoderInputIds.head.length * numAttnHeads * 64)(0.0f)
            .grouped(64)
            .toArray
            .grouped(decoderInputIds.head.length)
            .toArray
            .grouped(numAttnHeads)
            .toArray)

        val decoderInitFeedKeys = Map(
          "input_ids" -> decoderInputIdsTensors,
          "encoder_attention_mask" -> encoderAttentionMaskTensors,
          "encoder_hidden_states" -> encoderStateTensors,
          "use_cache_branch" -> OnnxTensor.createTensor(
            decoderEnv,
            Array(x = false))) ++ (encoderCacheInputKeys.map(x => (x, dummyTensor1))
          ++ decoderCacheInputKeys.map(x => (x, dummyTensor2)))

        val decoderInitFetchKeys =
          Array("logits") ++ encoderCacheOutputKeys ++ decoderCacheOutputKeys
        decoderInitResults = decoderSession.run(
          mapAsJavaMap(decoderInitFeedKeys),
          setAsJavaSet(Set(decoderInitFetchKeys).flatten))

        val decoderOutput = decoderInitResults
          .get(decoderInitFetchKeys.head)
          .get
          .asInstanceOf[OnnxTensor]
          .getFloatBuffer
          .array()
        decoderInitCache = Some(decoderInitFetchKeys.tail.map(cacheKey =>
          decoderInitResults.get(cacheKey).get.asInstanceOf[OnnxTensor]))
        decoderInputIdsTensors.close()

        decoderOutput
      } else {
        // Subsequent passes of the decoder
        val decoderInputIdsTensors =
          OnnxTensor.createTensor(decoderEnv, decoderInputIds.map(x => Array(x.last.toLong)))
        val decoderFeedKeys = Map(
          "input_ids" -> decoderInputIdsTensors,
          "encoder_attention_mask" -> encoderAttentionMaskTensors,
          "encoder_hidden_states" -> encoderStateTensors,
          "use_cache_branch" -> OnnxTensor.createTensor(
            decoderEnv,
            Array(x = true))) ++ encoderCacheInputKeys.zip(
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
        val decoderFetchKeys = Array("logits") ++ decoderCacheOutputKeys
        decoderResults = decoderSession.run(
          mapAsJavaMap(decoderFeedKeys),
          setAsJavaSet(Set(decoderFetchKeys).flatten))
        val decoderOutput = decoderResults
          .get(decoderFetchKeys.head)
          .get
          .asInstanceOf[OnnxTensor]
          .getFloatBuffer
          .array()
        decoderOutputCache = Some(decoderFetchKeys.tail.map(cacheKey =>
          decoderResults.get(cacheKey).get.asInstanceOf[OnnxTensor]))
        decoderInputIdsTensors.close()

        decoderOutput
      }

      val logits = (0 until batchSize)
        .map(i => {
          logitsRaw.slice(i * vocabSize, (i + 1) * vocabSize)
        })
        .toArray

      decoderInputIds =
        decoderProcessor.processLogits(batchLogits = logits, decoderInputIds = decoderInputIds)

    }

    if (decoderInitResults != null) decoderInitResults.close()
    if (decoderResults != null) decoderResults.close()

    decoderInputIds.map(x => x.filter(y => y != eosTokenId && y != paddingTokenId))
  }
}
