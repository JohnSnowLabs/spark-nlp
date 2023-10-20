package com.johnsnowlabs.ml.ai.marian

import com.johnsnowlabs.ml.tensorflow.{TensorResources, TensorflowWrapper}
import com.johnsnowlabs.ml.tensorflow.sentencepiece.SentencePieceWrapper
import com.johnsnowlabs.ml.tensorflow.sign.{ModelSignatureConstants, ModelSignatureManager}

import scala.collection.JavaConverters._

private[johnsnowlabs] class TensorflowMarianEncoderDecoder(
    val tensorflow: TensorflowWrapper,
    override val sppSrc: SentencePieceWrapper,
    override val sppTrg: SentencePieceWrapper,
    configProtoBytes: Option[Array[Byte]] = None,
    signatures: Option[Map[String, String]] = None)
    extends MarianEncoderDecoder(sppSrc, sppTrg) {

  val _tfMarianSignatures: Map[String, String] =
    signatures.getOrElse(ModelSignatureManager.apply())

  sessionWarmup()

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

    // Run encoder
    val tensorEncoder = new TensorResources()
    val inputDim = batch.length * maxSentenceLength

    val encoderInputIdsBuffers = tensorEncoder.createIntBuffer(batch.length * maxSentenceLength)
    val encoderAttentionMaskBuffers =
      tensorEncoder.createIntBuffer(batch.length * maxSentenceLength)
    val decoderAttentionMaskBuffers =
      tensorEncoder.createIntBuffer(batch.length * maxSentenceLength)

    val shape = Array(batch.length.toLong, maxSentenceLength)

    batch.zipWithIndex.foreach { case (tokenIds, idx) =>
      // this one marks the beginning of each sentence in the flatten structure
      val offset = idx * maxSentenceLength
      val diff = maxSentenceLength - tokenIds.length

      val s = tokenIds.take(maxSentenceLength) ++ Array.fill[Int](diff)(paddingTokenId)
      encoderInputIdsBuffers.offset(offset).write(s)
      val mask = s.map(x => if (x != paddingTokenId) 1 else 0)
      encoderAttentionMaskBuffers.offset(offset).write(mask)
      decoderAttentionMaskBuffers.offset(offset).write(mask)
    }

    val encoderInputIdsTensors =
      tensorEncoder.createIntBufferTensor(shape, encoderInputIdsBuffers)
    val encoderAttentionMaskKeyTensors =
      tensorEncoder.createIntBufferTensor(shape, encoderAttentionMaskBuffers)
    val decoderAttentionMaskTensors =
      tensorEncoder.createIntBufferTensor(shape, decoderAttentionMaskBuffers)

    val session = tensorflow.getTFSessionWithSignature(
      configProtoBytes = configProtoBytes,
      initAllTables = false,
      savedSignatures = signatures)
    val runner = session.runner

    runner
      .feed(
        _tfMarianSignatures
          .getOrElse(ModelSignatureConstants.EncoderInputIds.key, "missing_encoder_input_ids"),
        encoderInputIdsTensors)
      .feed(
        _tfMarianSignatures.getOrElse(
          ModelSignatureConstants.EncoderAttentionMask.key,
          "missing_encoder_attention_mask"),
        encoderAttentionMaskKeyTensors)
      .fetch(_tfMarianSignatures
        .getOrElse(ModelSignatureConstants.EncoderOutput.key, "missing_last_hidden_state"))

    val encoderOuts = runner.run().asScala
    val encoderOutsFloats = TensorResources.extractFloats(encoderOuts.head)
    val dim = encoderOutsFloats.length / inputDim
    val encoderOutsBatch =
      encoderOutsFloats.grouped(dim).toArray.grouped(maxSentenceLength).toArray

    encoderOuts.foreach(_.close())
    tensorEncoder.clearSession(encoderOuts)

    // Run decoder
    val decoderEncoderStateBuffers =
      tensorEncoder.createFloatBuffer(batch.length * maxSentenceLength * dim)
    batch.zipWithIndex.foreach { case (_, index) =>
      var offset = index * maxSentenceLength * dim
      encoderOutsBatch(index).foreach(encoderOutput => {
        decoderEncoderStateBuffers.offset(offset).write(encoderOutput)
        offset += dim
      })
    }

    val decoderEncoderStateTensors = tensorEncoder.createFloatBufferTensor(
      Array(batch.length.toLong, maxSentenceLength, dim),
      decoderEncoderStateBuffers)

    var decoderInputs = batch.map(_ => Array(paddingTokenId)).toArray
    var modelOutputs = batch.map(_ => Array(paddingTokenId)).toArray

    var stopDecoder = false

    while (!stopDecoder) {

      val decoderInputLength = decoderInputs.head.length
      val tensorDecoder = new TensorResources()

      val decoderInputBuffers = tensorDecoder.createIntBuffer(batch.length * decoderInputLength)

      decoderInputs.zipWithIndex.foreach { case (pieceIds, idx) =>
        val offset = idx * decoderInputLength
        decoderInputBuffers.offset(offset).write(pieceIds)
      }

      val decoderInputTensors = tensorDecoder.createIntBufferTensor(
        Array(batch.length.toLong, decoderInputLength),
        decoderInputBuffers)

      val runner = session.runner

      runner
        .feed(
          _tfMarianSignatures.getOrElse(
            ModelSignatureConstants.DecoderEncoderInputIds.key,
            "missing_encoder_state"),
          decoderEncoderStateTensors)
        .feed(
          _tfMarianSignatures
            .getOrElse(ModelSignatureConstants.DecoderInputIds.key, "missing_decoder_input_ids"),
          decoderInputTensors)
        .feed(
          _tfMarianSignatures.getOrElse(
            ModelSignatureConstants.DecoderAttentionMask.key,
            "missing_encoder_attention_mask"),
          decoderAttentionMaskTensors)
        .fetch(_tfMarianSignatures
          .getOrElse(ModelSignatureConstants.DecoderOutput.key, "missing_output_0"))

      val decoderOuts = runner.run().asScala
      val decoderOutputs = TensorResources
        .extractFloats(decoderOuts.head)
        .grouped(vocabSize)
        .toArray
        .grouped(decoderInputLength)
        .toArray

      val outputIds = decoderOutputs.map(batch =>
        batch
          .map(input => {
            var maxArg = -1
            var maxValue = Float.MinValue
            input.indices.foreach(i => {
              if ((input(i) >= maxValue) && (!ignoreTokenIds.contains(i))) {
                maxArg = i
                maxValue = input(i)
              }
            })
            maxArg
          })
          .last)
      decoderInputs = decoderInputs.zip(outputIds).map(x => x._1 ++ Array(x._2))
      modelOutputs = modelOutputs
        .zip(outputIds)
        .map(x => {
          if (x._1.contains(eosTokenId)) {
            x._1
          } else {
            x._1 ++ Array(x._2)
          }
        })

      decoderOuts.foreach(_.close())

      tensorDecoder.clearTensors()
      tensorDecoder.clearSession(decoderOuts)
      decoderInputTensors.close()

      stopDecoder = !modelOutputs.exists(o => o.last != eosTokenId) ||
        (modelOutputs.head.length > math.max(maxOutputLength, maxSentenceLength))

    }

    decoderAttentionMaskTensors.close()
    decoderEncoderStateTensors.close()
    tensorEncoder.clearTensors()

    modelOutputs.map(x => x.filter(y => y != eosTokenId && y != paddingTokenId))
  }
}
