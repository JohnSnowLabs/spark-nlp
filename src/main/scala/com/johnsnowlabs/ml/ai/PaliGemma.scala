/*
 * Copyright 2017-2024 John Snow Labs
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

import breeze.optimize.BatchSize
import com.johnsnowlabs.ml.ai.util.Generation.GenerationConfig
import com.johnsnowlabs.ml.onnx.OnnxWrapper.DecoderWrappers
import com.johnsnowlabs.ml.openvino.OpenvinoWrapper.PaliGemmaWrappers
import com.johnsnowlabs.nlp.annotators.common.Sentence
import com.johnsnowlabs.ml.util.{ONNX, Openvino}
import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common.SentenceSplit
import com.johnsnowlabs.nlp.annotators.cv.util.transform.ImageResizeUtils

import com.johnsnowlabs.nlp.annotators.cv.feature_extractor.Preprocessor
import com.johnsnowlabs.nlp.annotators.cv.util.io.ImageIOUtils
import com.johnsnowlabs.nlp.annotators.tokenizer.bpe.{
  BpeTokenizer,
  PaliGemmaTokenizer,
  SpecialTokens
}
import org.intel.openvino.InferRequest

import scala.collection.JavaConverters._

private[johnsnowlabs] class PaliGemma(
    val onnxWrappers: Option[DecoderWrappers],
    val openvinoWrapper: Option[PaliGemmaWrappers],
    merges: Map[(String, String), Int],
    vocabulary: Map[String, Int],
    addedTokens: Map[String, Int],
    preprocessor: Preprocessor,
    generationConfig: GenerationConfig,
    imageTokenLength: Int,
    imageToken: Int)
    extends Serializable {

  val detectedEngine: String =
    if (onnxWrappers.isDefined) ONNX.name
    else if (openvinoWrapper.isDefined) Openvino.name
    else Openvino.name

  private val GenerationConfig(
    bosTokenId: Int,
    paddingTokenId: Int,
    eosTokenId: Int,
    vocabSize: Int,
    beginSuppressTokens,
    suppressTokenIds,
    forcedDecoderIds) =
    generationConfig
  val reversedVocabulary: Map[Int, String] = vocabulary.map(_.swap)

  val specialTokens: SpecialTokens = SpecialTokens(
    vocabulary,
    startTokenString = reversedVocabulary(bosTokenId),
    endTokenString = reversedVocabulary(eosTokenId),
    unkTokenString = reversedVocabulary(eosTokenId),
    maskTokenString = reversedVocabulary(eosTokenId),
    padTokenString = reversedVocabulary(paddingTokenId),
    additionalStrings = addedTokens.keys.toArray)

  val bpeTokenizer: PaliGemmaTokenizer = BpeTokenizer
    .forModel(
      "paligemma",
      merges = merges,
      vocab = vocabulary,
      specialTokens = Some(specialTokens),
      addPrefixSpaceToSentence = false,
      alwaysAddPrefix = false,
      prependString = "")
    .asInstanceOf[PaliGemmaTokenizer]

  /** Decode a sequence of sentences
    * @param sentences
    *   Sequence of sentences
    * @return
    *   Sequence of decoded sentences
    */
  def decode(sentences: Array[Array[Int]]): Seq[String] = {
    sentences.map(s => bpeTokenizer.decodeTokens(s.map(_.toInt)))
  }

  /** Encode a sequence of sentences
    * @param sentences
    *   Sequence of sentences
    * @return
    *   Sequence of encoded sentences
    */
  def encodeText(sentences: Seq[Annotation], imgTokenLen: List[Int]): Seq[Array[Int]] = {

    val pattern = raw"<image>".r

    // raise an error if the pattern is not found in the text
    if (pattern.findFirstIn(sentences.head.result).isEmpty) {
      throw new IllegalArgumentException("The pattern <image> is not found in the text")
    }

    // split the sentences into chunks based on the pattern and tokenize them
    val promptChunks = sentences
      .map(s => {
        val sentWithTask = s.result
        var offsetLength = 0
        pattern
          .split(sentWithTask)
          .zipWithIndex
          .map(s => {
            val sentenceWithTask = Sentence(
              content = s._1,
              start = offsetLength,
              end = offsetLength + s._1.length,
              index = s._2)
            offsetLength += s._1.length
            bpeTokenizer
              .tokenize(sentenceWithTask)
              .map(bpeTokenizer.encode)
              .flatMap(_.map(_.pieceId))
          })
      })

    // inject the image padding tokens of length imgTokenLen between the prompt chunks
    val tokens = promptChunks
      .zip(imgTokenLen)
      .map(s => {
        val (promptChunk, imgTokenLen) = s
        val imgPaddingTokens = Array.fill(imgTokenLen)(imageToken)
        val combinedChunks = promptChunk
          .map(_.toArray)
          .reduce(_ ++ imgPaddingTokens ++ _)
        Array(bosTokenId) ++ combinedChunks
      })

    tokens
  }

  def encode(
      imageAnnotations: Seq[AnnotationImage],
      sentences: Seq[Annotation],
      preprocessor: Preprocessor,
      imageTokenLength: Int = imageTokenLength)
      : (Seq[Array[Int]], Array[Array[Array[Array[Float]]]]) = {
    val preprocessedImages = encodeImage(imageAnnotations.toArray, preprocessor)
    val encodedText = encodeText(sentences, List(imageTokenLength)).toArray

    (encodedText, preprocessedImages)
  }

  def tag(
      batch: Seq[Array[Int]],
      images: Array[Array[Array[Array[Float]]]],
      minOutputLength: Int,
      maxOutputLength: Int,
      doSample: Boolean,
      temperature: Double,
      topK: Int,
      topP: Double,
      repetitionPenalty: Double,
      noRepeatNgramSize: Int,
      randomSeed: Option[Long],
      ignoreTokenIds: Array[Int] = Array(),
      beamSize: Int,
      maxInputLength: Int,
      stopTokenIds: Array[Int]): Array[Array[Int]] = {

    val pixelValues = images
    val ignoreTokenIdsInt = ignoreTokenIds
    val expandedDecoderInputsVals = batch
    val sequencesLength = expandedDecoderInputsVals.map(x => x.length).toArray
    val maxSentenceLength = sequencesLength.max
    val numReturn_sequences = 1

    var effectiveBatch_size = 1
    var effectiveBatch_mult = 1

    if (doSample) {
      effectiveBatch_size = expandedDecoderInputsVals.length * numReturn_sequences
      effectiveBatch_mult = numReturn_sequences
    } else {
      effectiveBatch_size = expandedDecoderInputsVals.length
      effectiveBatch_mult = 1
    }

    val inferRequestLanguageModel =
      openvinoWrapper.get.languageModel.getCompiledModel().create_infer_request()
    val inferRequestImageEncoder =
      openvinoWrapper.get.imageEncoder.getCompiledModel().create_infer_request()
    val inferRequestTextEmbeddings =
      openvinoWrapper.get.textEmbeddings.getCompiledModel().create_infer_request()
    val inferRequestModelMerger =
      openvinoWrapper.get.modelMerger.getCompiledModel().create_infer_request()

    val generatedIds = generateGreedy(
      batch.toArray,
      batch.toArray,
      pixelValues,
      maxOutputLength,
      inferRequestLanguageModel,
      inferRequestImageEncoder,
      inferRequestTextEmbeddings,
      inferRequestModelMerger)
    generatedIds
  }

  def generateGreedy(
      encoderInputIds: Array[Array[Int]],
      decoderInputIds: Array[Array[Int]],
      pixelValues: Array[Array[Array[Array[Float]]]],
      maxOutputLength: Int,
      inferRequestLanguageModel: InferRequest,
      inferRequestImageEncoder: InferRequest,
      inferRequestTextEmbeddings: InferRequest,
      inferRequestModelMerger: InferRequest): Array[Array[Int]] = {

    var generatedIds: Array[Array[Int]] = Array()
    var decoderInputIdsCopied = decoderInputIds
    while (!greedyGenerationFinished(generatedIds, eosTokenId, maxOutputLength)) {
      val decoderOutputs = getModelOutputs(
        encoderInputIds,
        decoderInputIdsCopied,
        pixelValues,
        inferRequestLanguageModel,
        inferRequestImageEncoder,
        inferRequestTextEmbeddings,
        inferRequestModelMerger)

      val nextTokenIds = decoderOutputs.map { scores =>
        argmax(scores)
      }

      if (generatedIds.isEmpty) {
        generatedIds = nextTokenIds.map(Array(_))
      } else {
        generatedIds =
          generatedIds.zip(nextTokenIds).map { case (currentIds: Array[Int], nextId: Int) =>
            currentIds ++ Array(nextId)
          }
      }

      // extend decoder input ids
      decoderInputIdsCopied =
        decoderInputIdsCopied.zip(nextTokenIds).map { case (currentIds, nextId) =>
          currentIds ++ Array(nextId)
        }
    }
    generatedIds
  }

  def predict(
      sentences: Seq[Annotation],
      imageAnnotations: Seq[AnnotationImage],
      batchSize: Int,
      minOutputLength: Int,
      maxOutputLength: Int,
      doSample: Boolean,
      temperature: Double,
      topK: Int,
      topP: Double,
      repetitionPenalty: Double,
      noRepeatNgramSize: Int,
      randomSeed: Option[Long] = None,
      ignoreTokenIds: Array[Int] = Array(),
      beamSize: Int,
      maxInputLength: Int): Seq[Annotation] = {

    val (encodedText, preprocessedImages) = encode(imageAnnotations, sentences, preprocessor)
    val tagged = tag(
      encodedText,
      preprocessedImages,
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
      beamSize,
      maxInputLength,
      Array(eosTokenId))
    val decoded = decode(tagged)

    var sentBegin, nextSentEnd = 0
    val annotations = decoded.map { content =>
      nextSentEnd += content.length - 1
      val annots = new Annotation(
        annotatorType = DOCUMENT,
        begin = sentBegin,
        end = nextSentEnd,
        result = content,
        metadata = Map())
      sentBegin += nextSentEnd + 1
      annots
    }
    annotations
  }

  def getModelOutputs(
      encoderInputIds: Array[Array[Int]],
      decoderInputIds: Array[Array[Int]],
      pixelValues: Array[Array[Array[Array[Float]]]],
      inferRequestLanguageModel: InferRequest,
      inferRequestImageEncoder: InferRequest,
      inferRequestTextEmbeddings: InferRequest,
      inferRequestModelMerger: InferRequest): Array[Array[Float]] = {

    val inputEmbeds = getMultimodalEmbeddings(
      encoderInputIds,
      decoderInputIds,
      pixelValues,
      inferRequestImageEncoder,
      inferRequestTextEmbeddings,
      inferRequestModelMerger)

    val (inputIdsLong, inputPositionIDsLong): (Array[Long], Array[Long]) =
      if (encoderInputIds.head.length == decoderInputIds.head.length) {
        // First pass
        val inpIdsLong = decoderInputIds.flatMap { tokenIds => tokenIds.map(_.toLong) }
        val posIdsLong = decoderInputIds.flatMap { tokenIds =>
          tokenIds.zipWithIndex.map { case (_, i) =>
            i.toLong
          }
        }
        (inpIdsLong, posIdsLong)
      } else {
        // Subsequent passes
        val inpIdsLong = decoderInputIds.map { tokenIds => tokenIds.last.toLong }
        val posIdsLong = decoderInputIds.map { tokenIds =>
          tokenIds.zipWithIndex.map { case (_, i) =>
            i.toLong
          }.last
        }
        (inpIdsLong, posIdsLong)
      }
    val attentionMask: Array[Long] =
      decoderInputIds.flatMap { tokenIds => tokenIds.map(_ => 1L) }

    val batchSize: Int = decoderInputIds.length
    val beamIdx: Array[Int] = new Array[Int](batchSize)
    val shape: Array[Int] = Array(batchSize, inputIdsLong.length / batchSize)

    val decoderAttentionMask: org.intel.openvino.Tensor =
      new org.intel.openvino.Tensor(Array(batchSize, decoderInputIds.head.length), attentionMask)
    val decoderPositionIDs: org.intel.openvino.Tensor =
      new org.intel.openvino.Tensor(shape, inputPositionIDsLong)
    val beamIdxTensor: org.intel.openvino.Tensor =
      new org.intel.openvino.Tensor(Array(batchSize), beamIdx)

    inferRequestLanguageModel.set_tensor("inputs_embeds", inputEmbeds)
    inferRequestLanguageModel.set_tensor("attention_mask", decoderAttentionMask)
    inferRequestLanguageModel.set_tensor("position_ids", decoderPositionIDs)
    inferRequestLanguageModel.set_tensor("beam_idx", beamIdxTensor)

    inferRequestLanguageModel.infer()

    val result = inferRequestLanguageModel.get_tensor("logits")
    val logitsRaw = result.data()

    val sequenceLength = inputIdsLong.length / batchSize
    val decoderOutputs = (0 until batchSize).map(i => {
      logitsRaw
        .slice(
          i * sequenceLength * vocabSize + (sequenceLength - 1) * vocabSize,
          i * sequenceLength * vocabSize + sequenceLength * vocabSize)
    })
    decoderOutputs.toArray
  }

  private def argmax(scores: Array[Float]): Int =
    scores.zipWithIndex.maxBy { case (score, _) =>
      score
    }._2

  private def greedyGenerationFinished(
      decoderIds: Seq[Array[Int]],
      eosTokenId: Int,
      maxOutputLength: Int): Boolean = {
    if (decoderIds.isEmpty) {
      false
    } else {
      decoderIds.forall { ids =>
        ids.length >= maxOutputLength || ids.last == eosTokenId
      }
    }
  }

  private def encodeImage(
      annotations: Array[AnnotationImage],
      preprocessor: Preprocessor): Array[Array[Array[Array[Float]]]] = {

    val batchProcessedImages = annotations.map { annot =>
      val bufferedImage = ImageIOUtils.byteToBufferedImage(
        bytes = annot.result,
        w = annot.width,
        h = annot.height,
        nChannels = annot.nChannels)

      val resizedImage = if (preprocessor.do_resize) {
        ImageResizeUtils.resizeBufferedImage(
          width = preprocessor.size,
          height = preprocessor.size,
          preprocessor.resample)(bufferedImage)
      } else bufferedImage

      val normalizedImage =
        ImageResizeUtils.normalizeAndConvertBufferedImage(
          img = resizedImage,
          mean = preprocessor.image_mean,
          std = preprocessor.image_std,
          doNormalize = preprocessor.do_normalize,
          doRescale = preprocessor.do_rescale,
          rescaleFactor = preprocessor.rescale_factor)

      normalizedImage
    }

    batchProcessedImages
  }

  def getMultimodalEmbeddings(
      encoderInputIds: Array[Array[Int]],
      decoderInputIds: Array[Array[Int]],
      pixelValues: Array[Array[Array[Array[Float]]]],
      inferRequestImageEncoder: InferRequest,
      inferRequestTextEmbeddings: InferRequest,
      inferRequestModelMerger: InferRequest): org.intel.openvino.Tensor = {
    val inputIdsLong: Array[Long] =
      if (encoderInputIds.head.length == decoderInputIds.head.length) {
        // First pass
        val inpIdsLong = decoderInputIds.flatMap { tokenIds => tokenIds.map(_.toLong) }
        inpIdsLong
      } else {
        // Subsequent passes
        val inpIdsLong = decoderInputIds.map { tokenIds => tokenIds.last.toLong }
        inpIdsLong
      }
    val batchSize: Int = decoderInputIds.length
    val shape: Array[Int] = Array(batchSize, inputIdsLong.length / batchSize)
    val inputIdsLongTensor: org.intel.openvino.Tensor =
      new org.intel.openvino.Tensor(shape, inputIdsLong)

    val imageEmbeddings: org.intel.openvino.Tensor =
      if (encoderInputIds.head.length == decoderInputIds.head.length) {
        val pixelValuesTensor: org.intel.openvino.Tensor =
          new org.intel.openvino.Tensor(
            Array(batchSize, 3, 224, 224),
            pixelValues.flatten.flatten.flatten.map(_.toFloat))

        // Get image embeddings
        inferRequestImageEncoder.set_input_tensor(pixelValuesTensor)
        inferRequestImageEncoder.infer()
        val imageEmbeddings = inferRequestImageEncoder.get_output_tensor()

        // Get text embeddings
        inferRequestTextEmbeddings.set_input_tensor(inputIdsLongTensor)
        inferRequestTextEmbeddings.infer()
        val textEmbeddings = inferRequestTextEmbeddings.get_output_tensor()

        // Merge image and text embeddings
        inferRequestModelMerger.set_tensor("vision_embeds", imageEmbeddings)
        inferRequestModelMerger.set_tensor("inputs_embeds", textEmbeddings)
        inferRequestModelMerger.set_tensor("input_ids", inputIdsLongTensor)

        inferRequestModelMerger.infer()
        inferRequestModelMerger.get_tensor("final_embedding")
      } else {
        // Get text embeddings only for subsequent passes
        inferRequestTextEmbeddings.set_input_tensor(inputIdsLongTensor)
        inferRequestTextEmbeddings.infer()
        val textEmbeddings = inferRequestTextEmbeddings.get_output_tensor()
        textEmbeddings
      }
    imageEmbeddings
  }
}
