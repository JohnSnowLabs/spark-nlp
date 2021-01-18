package com.johnsnowlabs.ml.tensorflow

import com.johnsnowlabs.ml.tensorflow.sentencepiece._
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import com.johnsnowlabs.nlp.annotators.tokenizer.normalizer.MosesPunctNormalizer

import scala.collection.JavaConverters._

/** MarianTransformer: Fast Neural Machine Translation
  *
  * MarianTransformer uses models trained by MarianNMT.
  *
  * Marian is an efficient, free Neural Machine Translation framework written in pure C++ with minimal dependencies.
  * It is mainly being developed by the Microsoft Translator team. Many academic (most notably the University of Edinburgh and in the past the Adam Mickiewicz University in Poznań) and commercial contributors help with its development.
  *
  * It is currently the engine behind the Microsoft Translator Neural Machine Translation services and being deployed by many companies, organizations and research projects (see below for an incomplete list).
  *
  * '''Sources''' :
  * MarianNMT [[https://marian-nmt.github.io/]]
  * Marian: Fast Neural Machine Translation in C++ [[https://www.aclweb.org/anthology/P18-4020/]]
  *
  * @param tensorflow           LanguageDetectorDL Model wrapper with TensorFlow Wrapper
  * @param configProtoBytes     Configuration for TensorFlow session
  * @param sppSrc               Contains the vocabulary for the target language.
  * @param sppTrg               Contains the vocabulary for the source language
  */
class TensorflowMarian(val tensorflow: TensorflowWrapper,
                       val sppSrc: SentencePieceWrapper,
                       val sppTrg: SentencePieceWrapper,
                       configProtoBytes: Option[Array[Byte]] = None
                      ) extends Serializable {

  private val encoderInputIdsKey = "encoder_input_ids:0"
  private val encoderOutputsKey = "encoder_outputs:0"
  private val encoderAttentionMaskKey = "encoder_attention_mask:0"
  private val decoderInputIdsKey = "decoder_input_ids:0"
  private val decoderEncoderInputKey = "decoder_encoder_state:0"
  private val decoderAttentionMaskKey = "decoder_attention_mask:0"
  private val decoderPaddingMaskKey = "decoder_padding_mask:0"
  private val decoderCausalMaskKey = "decoder_causal_mask:0"
  private val decoderOutputsKey = "decoder_outputs:0"

  private val langCodeRe = ">>.+<<".r

  def process(batch: Seq[Array[Long]], maxOutputLength: Int, paddingTokenId: Long, eosTokenId: Long, vocabSize: Int): Array[Array[Long]] = {

    /* Actual size of each sentence to skip padding in the TF model */
    val sequencesLength = batch.map(x => x.length).toArray
    val maxSentenceLength = sequencesLength.max

    //Run encoder
    val tensorEncoderInputIds = new TensorResources()
    val tensorEncoderAttentionMask = new TensorResources()
    val tensorDecoderAttentionMask = new TensorResources()

    val encoderInputIdsBuffers = tensorEncoderInputIds.createLongBuffer(batch.length * maxSentenceLength)
    val encoderAttentionMaskBuffers = tensorEncoderAttentionMask.createLongBuffer(batch.length * maxSentenceLength)
    val decoderAttentionMaskBuffers = tensorDecoderAttentionMask.createLongBuffer(batch.length  * maxSentenceLength)

    val shape = Array(batch.length.toLong, maxSentenceLength)

    batch.foreach(tokenIds => {
      val diff = maxSentenceLength - tokenIds.length

      val s = tokenIds.take(maxSentenceLength) ++ Array.fill[Long](diff)(paddingTokenId)
      encoderInputIdsBuffers.put(s)
      val mask = s.map(x =>  if (x != paddingTokenId) 1L else 0L)
      encoderAttentionMaskBuffers.put(mask)
      decoderAttentionMaskBuffers.put(mask)
    })

    encoderInputIdsBuffers.flip()
    encoderAttentionMaskBuffers.flip()
    decoderAttentionMaskBuffers.flip()

    val encoderInputIdsTensors = tensorEncoderInputIds.createLongBufferTensor(shape, encoderInputIdsBuffers)
    val encoderAttentionMaskKeyTensors = tensorEncoderAttentionMask.createLongBufferTensor(shape, encoderAttentionMaskBuffers)
    val decoderAttentionMaskTensors = tensorDecoderAttentionMask.createLongBufferTensor(shape, decoderAttentionMaskBuffers)

    val session = tensorflow.getTFHubSession(configProtoBytes = configProtoBytes)
    val runner = session.runner

    runner
      .feed(encoderInputIdsKey, encoderInputIdsTensors)
      .feed(encoderAttentionMaskKey, encoderAttentionMaskKeyTensors)
      .fetch(encoderOutputsKey)

    val encoderOuts = runner.run().asScala
    val encoderOutputs = TensorResources.extractFloats(encoderOuts.head).grouped(512).toArray.grouped(maxSentenceLength).toArray

    encoderOuts.foreach(_.close())
    encoderInputIdsBuffers.clear()
    encoderAttentionMaskBuffers.clear()

    tensorEncoderInputIds.clearTensors()
    tensorEncoderAttentionMask.clearSession(encoderOuts)

    // Run decoder
    val tensorDecoderEncoderState = new TensorResources()
    val decoderEncoderStateBuffers = tensorDecoderEncoderState.createFloatBuffer(batch.length*maxSentenceLength*512)
    batch.zipWithIndex.foreach(bi => {
      encoderOutputs(bi._2).foreach(encoderOutput => {
        decoderEncoderStateBuffers.put(encoderOutput)
      })
    })
    decoderEncoderStateBuffers.flip()

    val decoderEncoderStateTensors = tensorDecoderEncoderState.createFloatBufferTensor(
      Array(batch.length.toLong, maxSentenceLength, 512),
      decoderEncoderStateBuffers)

    var decoderInputs = batch.map(_ => Array(paddingTokenId)).toArray
    var modelOutputs = batch.map(_ => Array(paddingTokenId)).toArray

    var stopDecoder = false

    while(!stopDecoder){

      val decoderInputLength = decoderInputs.head.length
      val tensorDecoderInput = new TensorResources()
      val tensorDecoderPaddingMask = new TensorResources()
      val tensorDecoderCasualMask = new TensorResources()

      val decoderInputBuffers = tensorDecoderInput.createLongBuffer(batch.length * decoderInputLength)
      val decoderPaddingMaskBuffers = tensorDecoderPaddingMask.createLongBuffer(batch.length * decoderInputLength)
      val decoderCasualMaskBuffers = if(decoderInputLength == 1) {
        tensorDecoderCasualMask.createFloatBuffer(1 * 1)
      }else{
        tensorDecoderCasualMask.createFloatBuffer(decoderInputLength * decoderInputLength)
      }

      decoderInputs.map{ pieceIds =>
        decoderInputBuffers.put(pieceIds)
        val paddingMasks = pieceIds.map(_ => if(pieceIds.length == 1) 1L else 0L)
        decoderPaddingMaskBuffers.put(paddingMasks)
      }

      if(decoderInputLength == 1){
        decoderCasualMaskBuffers.put(0.0f)
      }else{
        val casualMasks = Array.fill[Float](decoderInputLength*decoderInputLength)(0.0f)
        decoderCasualMaskBuffers.put(casualMasks)
      }

      decoderInputBuffers.flip()
      decoderPaddingMaskBuffers.flip()
      decoderCasualMaskBuffers.flip()

      val decoderInputTensors = tensorDecoderInput.createLongBufferTensor(
        Array(batch.length.toLong, decoderInputLength), decoderInputBuffers)
      val decoderPaddingMaskTensors = tensorDecoderPaddingMask.createLongBufferTensor(
        Array(batch.length.toLong, decoderInputLength), decoderPaddingMaskBuffers)
      val decoderCausalMaskTensors = if(decoderInputLength == 1) {
        tensorDecoderCasualMask.createFloatBufferTensor(Array(1L, 1), decoderCasualMaskBuffers)
      }else {
        tensorDecoderCasualMask.createFloatBufferTensor(
          Array(decoderInputLength.toLong, decoderInputLength), decoderCasualMaskBuffers)
      }

      val runner = session.runner

      runner
        .feed(decoderEncoderInputKey, decoderEncoderStateTensors)
        .feed(decoderInputIdsKey, decoderInputTensors)
        .feed(decoderAttentionMaskKey, decoderAttentionMaskTensors)
        .feed(decoderPaddingMaskKey, decoderPaddingMaskTensors)
        .feed(decoderCausalMaskKey, decoderCausalMaskTensors)
        .fetch(decoderOutputsKey)

      val decoderOuts = runner.run().asScala
      val decoderOutputs = TensorResources.extractFloats(decoderOuts.head)
        .grouped(vocabSize).toArray.grouped(decoderInputLength).toArray

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

      decoderInputBuffers.clear()
      decoderPaddingMaskBuffers.clear()
      decoderCasualMaskBuffers.clear()

      tensorDecoderInput.clearTensors()
      tensorDecoderPaddingMask.clearTensors()
      tensorDecoderCasualMask.clearTensors()

      tensorEncoderAttentionMask.clearSession(decoderOuts)

      stopDecoder = !modelOutputs.exists(o => o.last != eosTokenId) ||
        (modelOutputs.head.length > math.max(maxOutputLength, maxSentenceLength))

    }

    tensorEncoderInputIds.clearTensors()
    tensorEncoderAttentionMask.clearTensors()
    tensorDecoderEncoderState.clearTensors()
    decoderEncoderStateBuffers.clear()

    modelOutputs.map(x => x.filter(y => y != eosTokenId && y != paddingTokenId))
  }

  def decode(sentences: Array[Array[Long]], vocabsArray: Array[String]): Seq[String] = {

    sentences.map { s =>
      val filteredPads = s.filter(x => x != 0L)
      val pieceTokens = filteredPads.map {
        pieceId =>
          vocabsArray(pieceId.toInt)
      }
      sppTrg.getSppModel.decodePieces(pieceTokens.toList.asJava)
    }

  }

  def encode(sentences: Seq[Annotation], normalizer: MosesPunctNormalizer, maxSeqLength: Int, vocabsArray: Array[String],
             langId: Long, unknownTokenId: Long, eosTokenId: Long): Seq[Array[Long]] = {

    sentences.map { s =>
      // remove langauge code from the source text
      val sentWithouLangId = langCodeRe.replaceFirstIn(s.result, "").trim
      val normalizedSent = normalizer.normalize(sentWithouLangId)
      val pieceTokens = sppSrc.getSppModel.encodeAsPieces(normalizedSent).toArray.map(x=>x.toString)

      val pieceIds = pieceTokens.map {
        piece =>
          val pieceId = vocabsArray.indexOf(piece).toLong
          if (pieceId > 0L) {
            pieceId
          } else {
            unknownTokenId
          }
      }

      if(langId > 0L)
        Array(langId) ++ pieceIds.take(maxSeqLength) ++ Array(eosTokenId)
      else
        pieceIds.take(maxSeqLength) ++ Array(eosTokenId)
    }

  }

  /*
  * Batch size more than 1 is not performing well on CPU
  *
  * */
  def generateSeq2Seq(sentences: Seq[Annotation],
                      batchSize: Int = 1,
                      maxInputLength: Int,
                      maxOutputLength: Int,
                      vocabs: Array[String],
                      langId: String
                     ): Array[Annotation] = {

    val normalizer = new MosesPunctNormalizer()

    val paddingTokenId = vocabs.indexOf("<pad>").toLong
    val unknownTokenId = vocabs.indexOf("<unk>").toLong
    val eosTokenId = vocabs.indexOf("</s>").toLong
    val vocabSize = vocabs.toSeq.length

    val langIdPieceId = if (langId.nonEmpty) {
      vocabs.indexOf(langId).toLong
    } else {
      val lang = langCodeRe.findFirstIn(sentences.head.result.trim).getOrElse(-1L)
      vocabs.indexOf(lang).toLong
    }

    val batchDecoder = sentences.grouped(batchSize).toArray.flatMap { batch =>

      val batchSP = encode(batch, normalizer, maxInputLength, vocabs, langIdPieceId, unknownTokenId, eosTokenId)
      val spIds = process(batchSP,maxOutputLength, paddingTokenId, eosTokenId, vocabSize)
      decode(spIds, vocabs)

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
