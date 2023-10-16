package com.johnsnowlabs.ml.ai.t5

import ai.onnxruntime.{OnnxTensor, OrtSession, TensorInfo}
import com.johnsnowlabs.ml.onnx.OnnxWrapper
import com.johnsnowlabs.ml.tensorflow.sentencepiece.SentencePieceWrapper
import org.apache.hadoop.thirdparty.org.checkerframework.checker.units.qual.Temperature

import scala.collection.JavaConverters.{mapAsJavaMap, setAsJavaSet}

class OnnxT5EncoderDecoder(
                                   val onnxEncoder: OnnxWrapper,
                                   val onnxDecoder: OnnxWrapper,
                                   override val spp: SentencePieceWrapper,
                                   override val additionalTokens: Map[Int, String] = Map()
                                 )
  extends T5EncoderDecoder(spp, additionalTokens) {

  protected val numLayers: Int = {
    ((onnxDecoder.getSession()._1.getNumOutputs - 1) / 4).toInt
  }

  protected val numAttnHeads: Int = {
    onnxDecoder
      .getSession()._1
      .getInputInfo
      .get("past_key_values.0.decoder.value")
      .getInfo.asInstanceOf[TensorInfo].getShape()(1).toInt
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
    /* Actual size of each sentence to skip padding in the TF model */
    val sequencesLength = batch.map(x => x.length).toArray
    val maxSentenceLength = sequencesLength.max // - curLen

    val numReturn_sequences = 1
    // from config
    val vocabSize = 32128

    val inputDim = batch.length * maxSentenceLength

    val (encoder, env) = onnxEncoder.getSession()

    // Run encoder
    val encoderInputBuffers = batch.map(
      tokenIds => (tokenIds.take(maxSentenceLength).map(_.toLong) ++ Array.fill[Long](maxSentenceLength - tokenIds.length)(this.paddingTokenId))
    ).toArray
    val encoderAttentionMaskBuffers = encoderInputBuffers.map(x => x.map(xx => if (xx != this.paddingTokenId) 1L else 0L))

    val encoderInputTensors = OnnxTensor.createTensor(env, encoderInputBuffers)
    val encoderAttentionMaskTensors = OnnxTensor.createTensor(env, encoderAttentionMaskBuffers)

    Array(Array())
    try {

      val encoderResults = encoder.run(mapAsJavaMap(Map(
        "input_ids" -> encoderInputTensors,
        "attention_mask" -> encoderAttentionMaskTensors
      )))

      val encoderStateBuffer = try {
        val encoderStateTensor = encoderResults
          .get("last_hidden_state")
          .get()
          .asInstanceOf[OnnxTensor]

        val shape = encoderStateTensor.getInfo.getShape
        encoderStateTensor
          .getFloatBuffer
          .array()
          .grouped(shape(2).toInt).toArray.grouped(shape(1).toInt).toArray
      } finally {
        if (encoderResults != null) encoderResults.close()
      }

      encoderInputTensors.close()

      val encoderStateTensors = OnnxTensor.createTensor(env, encoderStateBuffer)

      val modelOutputs = generateNoBeamSearch(
        batch,
        encoderStateTensors,
        encoderAttentionMaskTensors,
        maxNewTokens=maxNewTokens,
        maxTextLength=maxTextLength,
        doSample=doSample,
        topK=topK,
        topP=topP,
        temperature=temperature,
        vocabSize=vocabSize,
        randomSeed=randomSeed,
        ignoreTokenIds=ignoreTokenIds,
        stopAtEos=stopAtEos,
        noRepeatNgramSize=noRepeatNgramSize,
        repetitionPenalty = repetitionPenalty)

      encoderAttentionMaskTensors.close()
      encoderStateTensors.close()
      //      modelOutputs.foreach(x => {
      //        println(x.map(_.toString).mkString(" "))
      //      })
      modelOutputs
    }
  }

  def generateCacheKeys(component: String, state: String): Array[String] = {
    {0 until numLayers}.flatMap(x => Array(s"$state.$x.$component.key", s"$state.$x.$component.value")).toArray
  }

  lazy val encoderCacheInputKeys: Array[String] = generateCacheKeys("encoder", "past_key_values")
  lazy val encoderCacheOutputKeys: Array[String] = generateCacheKeys("encoder", "present")
  lazy val decoderCacheInputKeys: Array[String] = generateCacheKeys("decoder", "past_key_values")
  lazy val decoderCacheOutputKeys: Array[String] = generateCacheKeys("decoder", "present")

  def generateNoBeamSearch(
                            inputIds: Seq[Array[Int]],
                            encoderStateTensors: OnnxTensor,
                            encoderAttentionMaskTensors: OnnxTensor,
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
    val stopTokens = if (stopAtEos) Array(this.eosTokenId) else Array[Int]()
    var decoderInitCache: Option[Array[OnnxTensor]] = None
    var decoderOutputCache: Option[Array[OnnxTensor]] = None
    var decoderInitResults: OrtSession.Result = null
    var decoderResults: OrtSession.Result = null
    val (decoderSession, decoderEnv) = onnxDecoder.getSession()

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
      repetitionPenalty = repetitionPenalty
    )

    while (!decoderProcessor.stopDecoding(decoderInputIds)) {

      var logitsRaw: Array[Float] = Array()
      if (decoderInitCache.isEmpty) {
        //First pass of the decoder
        val decoderInputIdsTensors = OnnxTensor.createTensor(decoderEnv, decoderInputIds)
        //dummy zero tensors for the first pass
        val dummyTensor1 = OnnxTensor.createTensor(decoderEnv,
          Array.fill(batchSize * inputIds.head.length * numAttnHeads * 64)(0.0f)
            .grouped(64)
            .toArray
            .grouped(inputIds.head.length)
            .toArray
            .grouped(numAttnHeads)
            .toArray)
        val dummyTensor2 = OnnxTensor.createTensor(decoderEnv,
          Array.fill(batchSize * decoderInputIds.head.length * numAttnHeads * 64)(0.0f)
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
          "use_cache_branch" -> OnnxTensor.createTensor(decoderEnv, Array(x = false))
        ) ++ (
          encoderCacheInputKeys.map(x => (x, dummyTensor1))
            ++ decoderCacheInputKeys.map(x => (x, dummyTensor2)))

        val decoderInitFetchKeys = Array("logits") ++ encoderCacheOutputKeys ++ decoderCacheOutputKeys
        decoderInitResults = decoderSession.run(mapAsJavaMap(decoderInitFeedKeys), setAsJavaSet(Set(decoderInitFetchKeys).flatten))
        logitsRaw = decoderInitResults
          .get(decoderInitFetchKeys.head)
          .get
          .asInstanceOf[OnnxTensor]
          .getFloatBuffer
          .array()
        decoderInitCache = Some(
          decoderInitFetchKeys.tail.map(cacheKey => decoderInitResults.get(cacheKey).get.asInstanceOf[OnnxTensor]))
        decoderInputIdsTensors.close()
      } else {
        //Subsequent passes of the decoder
        val decoderInputIdsTensors = OnnxTensor.createTensor(decoderEnv, decoderInputIds.map(x => Array(x.last)))
        val decoderFeedKeys = Map(
          "input_ids" -> decoderInputIdsTensors,
          "encoder_attention_mask" -> encoderAttentionMaskTensors,
          "encoder_hidden_states" -> encoderStateTensors,
          "use_cache_branch" -> OnnxTensor.createTensor(decoderEnv, Array(x = true))
        ) ++ encoderCacheInputKeys.zip(decoderInitCache.get.slice(0, encoderCacheInputKeys.length)) ++ (
          if (decoderOutputCache.isEmpty){
            decoderCacheInputKeys.zip(decoderInitCache.get.slice(encoderCacheInputKeys.length, encoderCacheInputKeys.length + decoderCacheInputKeys.length))
          } else {
            decoderCacheInputKeys.zip(decoderOutputCache.get)
          }
          )
        val decoderFetchKeys = Array("logits") ++ decoderCacheOutputKeys
        decoderResults = decoderSession.run(mapAsJavaMap(decoderFeedKeys), setAsJavaSet(Set(decoderFetchKeys).flatten))
        logitsRaw = decoderResults
          .get(decoderFetchKeys.head)
          .get
          .asInstanceOf[OnnxTensor]
          .getFloatBuffer
          .array()
        decoderOutputCache = Some(decoderFetchKeys.tail.map(cacheKey => decoderResults.get(cacheKey).get.asInstanceOf[OnnxTensor]))
        decoderInputIdsTensors.close()
      }

      val logits  = (0 until batchSize).map(i => {
        logitsRaw.slice(i * vocabSize, (i + 1) * vocabSize)
      }).toArray

      decoderInputIds = decoderProcessor.processLogits(batchLogits = logits, decoderInputIds = decoderInputIds)

    }
    if (decoderInitResults != null) decoderInitResults.close()
    if (decoderResults != null) decoderResults.close()
    decoderInputIds.map(x => x.map(_.toInt))
  }
}
