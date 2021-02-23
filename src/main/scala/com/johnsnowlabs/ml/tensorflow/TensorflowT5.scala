package com.johnsnowlabs.ml.tensorflow

import com.johnsnowlabs.ml.tensorflow.sentencepiece._
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}

import scala.collection.JavaConverters._

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
  private val decoderEncoderAttentionMaskKey = "decoder_encoder_attention_mask:0"
  private val decoderAttentionMaskKey = "decoder_attention_mask:0"

  private val encoderOutputsKey = "encoder_outputs:0"
  private val decoderOutputsKey = "decoder_outputs:0"

  private val paddingTokenId = 0L
  private val eosTokenId = 1L
  private val pieceSize = spp.getSppModel.getPieceSize


  def process(batch: Seq[Array[Long]], maxOutputLength: Int = 200): Array[Array[Long]] = {

    /* Actual size of each sentence to skip padding in the TF model */
    val sequencesLength = batch.map(x => x.length).toArray
    val maxSentenceLength = sequencesLength.max

    //Run encoder
    val tensorEncoder = new TensorResources()
    val inputDim = batch.length * maxSentenceLength

    val encoderInputBuffers = tensorEncoder.createLongBuffer(inputDim)
    val encoderAttentionMaskBuffers = tensorEncoder.createLongBuffer(inputDim)

    val shape = Array(batch.length.toLong, maxSentenceLength)

    batch.zipWithIndex.foreach { case (tokenIds, idx) =>
      val offset = idx * maxSentenceLength
      val diff = maxSentenceLength - tokenIds.length

      val s = tokenIds.take(maxSentenceLength) ++ Array.fill[Long](diff)(this.paddingTokenId)
      encoderInputBuffers.offset(offset).write(s)
      val mask = s.map(x => if (x != this.paddingTokenId) 1L else 0L)
      encoderAttentionMaskBuffers.offset(offset).write(mask)
    }

    val encoderInputTensors = tensorEncoder.createLongBufferTensor(shape, encoderInputBuffers)
    val encoderAttentionMaskTensors = tensorEncoder.createLongBufferTensor(shape, encoderAttentionMaskBuffers)

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

    encoderOuts.foreach(_.close())
    tensorEncoder.clearSession(encoderOuts)

    //Run decoder
    val decoderEncoderStateBuffers = tensorEncoder.createFloatBuffer(batch.length * maxSentenceLength * dim)
    //TODO: There must be a better way to calculate offsets
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

    var decoderInputs = batch.map(_ => Array(this.paddingTokenId)).toArray
    var modelOutputs = batch.map(_ => Array(this.paddingTokenId)).toArray

    var stopDecoder = false

    while (!stopDecoder) {
      val decoderInputLength = decoderInputs.head.length
      val tensorDecoder = new TensorResources()

      val decoderInputBuffers = tensorDecoder.createLongBuffer(batch.length * decoderInputLength)
      val decoderAttentionBuffers = tensorDecoder.createLongBuffer(batch.length * decoderInputLength)

      decoderInputs.zipWithIndex.foreach{ case (pieceIds, idx) =>
        val offset = idx * decoderInputLength
        decoderInputBuffers.offset(offset).write(pieceIds)
        val paddingMasks = pieceIds.map(_ => 1L)
        decoderAttentionBuffers.offset(offset).write(paddingMasks)
      }

      val decoderInputTensors = tensorDecoder.createLongBufferTensor(
        Array(batch.length.toLong, decoderInputLength), decoderInputBuffers)
      val decoderAttentionMaskTensors = tensorDecoder.createLongBufferTensor(
        Array(batch.length.toLong, decoderInputLength), decoderAttentionBuffers)
      val runner = session.runner

      runner
        .feed(decoderInputIdsKey, decoderInputTensors)
        .feed(decoderEncoderStateKey, decoderEncoderStateTensors)
        .feed(decoderEncoderAttentionMaskKey, encoderAttentionMaskTensors)
        .feed(decoderAttentionMaskKey, decoderAttentionMaskTensors)
        .fetch(decoderOutputsKey)

      val decoderOuts = runner.run().asScala

      val decoderOutputs = TensorResources.extractFloats(decoderOuts.head).grouped(32128).toArray.grouped(decoderInputLength).toArray

      val outputIds = decoderOutputs.map(batch => batch.map(input => input.indexOf(input.max)).last).map(_.toLong)
      decoderInputs = decoderInputs.zip(outputIds).map(x => x._1 ++ Array(x._2))
      modelOutputs = modelOutputs.zip(outputIds).map(x => {
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

      stopDecoder = (
        !modelOutputs.exists(o => o.last != this.eosTokenId)
          || (modelOutputs.head.length > maxOutputLength))

    }

    tensorEncoder.clearTensors()
    modelOutputs
  }

  def decode(sentences: Array[Array[Long]]): Seq[String] = {

    sentences.map { s =>
      val filteredPieceIds = s.filter(x => x <= pieceSize)
      spp.getSppModel.decodeIds(filteredPieceIds.map(_.toInt):_*)
    }

  }

  def encode(sentences: Seq[Annotation], task: String): Seq[Array[Long]] = {
    sentences.map(
      s => {
        val sentWithTask = if(task.nonEmpty) task.concat(" ").concat(s.result) else s.result
        spp.getSppModel.encodeAsIds(sentWithTask).map(_.toLong) ++ Array(this.eosTokenId)
      })
  }

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
    batchDecoder.zip(sentences).map{
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

}