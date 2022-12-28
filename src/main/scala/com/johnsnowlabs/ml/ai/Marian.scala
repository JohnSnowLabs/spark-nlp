/*
 * Copyright 2017-2022 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.ml.ai

import com.johnsnowlabs.ml.ai.util.sentencepiece.SentencePieceWrapper
import com.johnsnowlabs.ml.tensorflow.sign.{ModelSignatureConstants, ModelSignatureManager}
import com.johnsnowlabs.ml.tensorflow.{TensorResources, TensorflowWrapper}
import com.johnsnowlabs.nlp.annotators.tokenizer.normalizer.MosesPunctNormalizer
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}

import scala.collection.JavaConverters._

/** MarianTransformer: Fast Neural Machine Translation
  *
  * MarianTransformer uses models trained by MarianNMT.
  *
  * Marian is an efficient, free Neural Machine Translation framework written in pure C++ with
  * minimal dependencies. It is mainly being developed by the Microsoft Translator team. Many
  * academic (most notably the University of Edinburgh and in the past the Adam Mickiewicz
  * University in Poznań) and commercial contributors help with its development.
  *
  * It is currently the engine behind the Microsoft Translator Neural Machine Translation services
  * and being deployed by many companies, organizations and research projects (see below for an
  * incomplete list).
  *
  * '''Sources''' : MarianNMT [[https://marian-nmt.github.io/]] Marian: Fast Neural Machine
  * Translation in C++ [[https://www.aclweb.org/anthology/P18-4020/]]
  *
  * @param tensorflow
  *   LanguageDetectorDL Model wrapper with TensorFlow Wrapper
  * @param configProtoBytes
  *   Configuration for TensorFlow session
  * @param sppSrc
  *   Contains the vocabulary for the target language.
  * @param sppTrg
  *   Contains the vocabulary for the source language
  */
private[johnsnowlabs] class Marian(
    val tensorflow: TensorflowWrapper,
    val sppSrc: SentencePieceWrapper,
    val sppTrg: SentencePieceWrapper,
    configProtoBytes: Option[Array[Byte]] = None,
    signatures: Option[Map[String, String]] = None)
    extends Serializable {

  val _tfMarianSignatures: Map[String, String] =
    signatures.getOrElse(ModelSignatureManager.apply())

  private val langCodeRe = ">>.+<<".r

  private def sessionWarmup(): Unit = {
    val dummyInput = Array.fill(1)(0)
    tag(Seq(dummyInput), 0, 0, 0, 1)
  }

  sessionWarmup()

  def tag(
      batch: Seq[Array[Int]],
      maxOutputLength: Int,
      paddingTokenId: Int,
      eosTokenId: Int,
      vocabSize: Int,
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

  def decode(sentences: Array[Array[Int]], vocabsArray: Array[String]): Seq[String] = {

    sentences.map { s =>
      val filteredPads = s.filter(x => x != 0)
      val pieceTokens = filteredPads.map { pieceId =>
        vocabsArray(pieceId)
      }
      sppTrg.getSppModel.decodePieces(pieceTokens.toList.asJava)
    }

  }

  def encode(
      sentences: Seq[Annotation],
      normalizer: MosesPunctNormalizer,
      maxSeqLength: Int,
      vocabsArray: Array[String],
      langId: Int,
      unknownTokenId: Int,
      eosTokenId: Int): Seq[Array[Int]] = {

    sentences.map { s =>
      // remove language code from the source text
      val sentWithoutLangId = langCodeRe.replaceFirstIn(s.result, "").trim
      val normalizedSent = normalizer.normalize(sentWithoutLangId)
      val pieceTokens =
        sppSrc.getSppModel.encodeAsPieces(normalizedSent).toArray.map(x => x.toString)

      val pieceIds = pieceTokens.map { piece =>
        val pieceId = vocabsArray.indexOf(piece)
        if (pieceId > 0) {
          pieceId
        } else {
          unknownTokenId
        }
      }

      if (langId > 0)
        Array(langId) ++ pieceIds.take(maxSeqLength - 2) ++ Array(eosTokenId)
      else
        pieceIds.take(maxSeqLength - 1) ++ Array(eosTokenId)
    }

  }

  /** generate seq2seq via encoding, generating, and decoding
    *
    * @param sentences
    *   none empty Annotation
    * @param batchSize
    *   size of baches to be process at the same time
    * @param maxInputLength
    *   maximum length for input
    * @param maxOutputLength
    *   maximum length for output
    * @param vocabs
    *   list of all vocabs
    * @param langId
    *   language id for multi-lingual models
    * @return
    */
  def predict(
      sentences: Seq[Annotation],
      batchSize: Int = 1,
      maxInputLength: Int,
      maxOutputLength: Int,
      vocabs: Array[String],
      langId: String,
      ignoreTokenIds: Array[Int] = Array()): Array[Annotation] = {

    val normalizer = new MosesPunctNormalizer()

    val paddingTokenId = vocabs.indexOf("<pad>")
    val unknownTokenId = vocabs.indexOf("<unk>")
    val eosTokenId = vocabs.indexOf("</s>")
    val ignoreTokenIdsWithPadToken = ignoreTokenIds ++ Array(paddingTokenId)
    val vocabSize = vocabs.toSeq.length

    val langIdPieceId = if (langId.nonEmpty) {
      vocabs.indexOf(langId)
    } else {
      val lang = langCodeRe.findFirstIn(sentences.head.result.trim).getOrElse(-1L)
      vocabs.indexOf(lang)
    }

    val batchDecoder = sentences.grouped(batchSize).toArray.flatMap { batch =>
      val batchSP = encode(
        batch,
        normalizer,
        maxInputLength,
        vocabs,
        langIdPieceId,
        unknownTokenId,
        eosTokenId)
      val spIds = tag(
        batchSP,
        maxOutputLength,
        paddingTokenId,
        eosTokenId,
        vocabSize,
        ignoreTokenIdsWithPadToken)
      decode(spIds, vocabs)

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

}
