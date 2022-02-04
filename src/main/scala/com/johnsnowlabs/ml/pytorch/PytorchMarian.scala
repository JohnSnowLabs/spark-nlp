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

package com.johnsnowlabs.ml.pytorch

import ai.djl.inference.Predictor
import ai.djl.{Device, Model}
import ai.djl.ndarray.NDList
import ai.djl.pytorch.engine.PtModel
import ai.djl.translate.{Batchifier, Translator, TranslatorContext}
import com.johnsnowlabs.ml.tensorflow.sentencepiece.SentencePieceWrapper
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import com.johnsnowlabs.nlp.annotators.tokenizer.normalizer.MosesPunctNormalizer

import scala.collection.JavaConverters._
import java.io.ByteArrayInputStream

class PytorchMarian (val pytorchWrapper: PytorchWrapper,
                     val sppSrc: SentencePieceWrapper,
                     val sppTrg: SentencePieceWrapper)
  extends Serializable with Translator[Array[Array[Int]], Array[Array[Int]]] {

  private val langCodeRe = ">>.+<<".r

  private def sessionWarmup(): Unit = {
    val dummyInput = Array.fill(1)(0)
    tag(Seq(dummyInput), 0, 0, 0, 1)
  }

  //TODO: Does it need to warmup as in TF version??
  //sessionWarmup()

  def predict(sentences: Seq[Annotation],
              batchSize: Int = 1,
              maxInputLength: Int,
              maxOutputLength: Int,
              vocabs: Array[String],
              langId: String,
              ignoreTokenIds: Array[Int] = Array()
             ): Array[Annotation] = {

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

      val batchSP = encode(batch, normalizer, maxInputLength, vocabs, langIdPieceId, unknownTokenId, eosTokenId)
      val spIds = tag(batchSP, maxOutputLength, paddingTokenId, eosTokenId, vocabSize, ignoreTokenIdsWithPadToken)
      decode(spIds, vocabs)

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

  def encode(sentences: Seq[Annotation], normalizer: MosesPunctNormalizer, maxSeqLength: Int, vocabsArray: Array[String],
             langId: Int, unknownTokenId: Int, eosTokenId: Int): Seq[Array[Int]] = {

    sentences.map { s =>
      // remove language code from the source text
      val sentWithoutLangId = langCodeRe.replaceFirstIn(s.result, "").trim
      val normalizedSent = normalizer.normalize(sentWithoutLangId)
      val pieceTokens = sppSrc.getSppModel.encodeAsPieces(normalizedSent).toArray.map(x => x.toString)

      val pieceIds = pieceTokens.map {
        piece =>
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

  def decode(sentences: Array[Array[Int]], vocabsArray: Array[String]): Seq[String] = {

    sentences.map { s =>
      val filteredPads = s.filter(x => x != 0)
      val pieceTokens = filteredPads.map {
        pieceId =>
          vocabsArray(pieceId)
      }
      sppTrg.getSppModel.decodePieces(pieceTokens.toList.asJava)
    }

  }

  protected lazy val predictor: Predictor[Array[Array[Int]], Array[Array[Int]]] = {
    val modelInputStream = new ByteArrayInputStream(pytorchWrapper.modelBytes)
    val device = Device.cpu() //TODO: Check with gpu
    val model = Model.newInstance("pytorch-model", device)

    val pyTorchModel: PtModel = model.asInstanceOf[PtModel]
    pyTorchModel.load(modelInputStream)

    pyTorchModel.newPredictor(this)
  }

  def tag(batch: Seq[Array[Int]],
          maxOutputLength: Int,
          paddingTokenId: Int,
          eosTokenId: Int,
          vocabSize: Int,
          ignoreTokenIds: Array[Int] = Array()): Array[Array[Int]] = {

    /* Actual size of each sentence to skip padding in the Pytorch model */
    val sequencesLength = batch.map(x => x.length).toArray
    val maxSentenceLength = sequencesLength.max

    //For testing
    val inputIds: Array[Int] = Array(3923, 2, 125, 6913, 31, 15873, 0)
    val attentionMask: Array[Int] = Array(1, 1, 1, 1, 1, 1, 1)
    val decoderInputIds: Array[Int] = Array(65000,  3923,     2,   125,  6913,    31, 15873)

    val dummyInput: Array[Array[Int]] = Array(inputIds, attentionMask, decoderInputIds)
    val dummyOutput = predictor.predict(dummyInput)

    val output = predictor.predict(batch.toArray)
    val dimension = output.head.head
    val encoderOutsFloats = output.last
    val encoderOutsBatch = encoderOutsFloats
      .grouped(dimension).toArray
      .grouped(maxSentenceLength).toArray

    Array()

//    var decoderInputs = batch.map(_ => Array(paddingTokenId)).toArray
//    var modelOutputs = batch.map(_ => Array(paddingTokenId)).toArray
//
//    var stopDecoder = false
//
//    while (!stopDecoder) {
//      val decoderInputLength = decoderInputs.head.length
//    }

  }

  override def getBatchifier: Batchifier = {
    Batchifier.fromString("none")
  }

  override def processInput(ctx: TranslatorContext, input: Array[Array[Int]]): NDList = {
    val manager = ctx.getNDManager
    val array = manager.create(input)
    new NDList(array)
  }

  override def processOutput(ctx: TranslatorContext, list: NDList): Array[Array[Int]] = {
    val dimension = Array(list.get(0).getShape.get(2).toInt)
    val allEncoderLayers = list.get(0).toIntArray

    Array(dimension, allEncoderLayers)
  }

}
