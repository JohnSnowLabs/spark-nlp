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

import com.johnsnowlabs.ml.ai.util.Generation.GenerationConfig
import com.johnsnowlabs.ml.onnx.OnnxWrapper.DecoderWrappers
import com.johnsnowlabs.ml.openvino.OpenvinoWrapper.E5VWrappers
import com.johnsnowlabs.ml.util.{ONNX, Openvino}
import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common.SentenceSplit
import com.johnsnowlabs.ml.ai.util.transform.E5VUtils
import com.johnsnowlabs.nlp.annotators.cv.feature_extractor.Preprocessor
import com.johnsnowlabs.nlp.annotators.cv.util.io.ImageIOUtils
import com.johnsnowlabs.nlp.annotators.cv.util.transform.ImageResizeUtils
import com.johnsnowlabs.nlp.annotators.tokenizer.bpe.{BpeTokenizer, LLAVATokenizer, SpecialTokens}
import org.intel.openvino.InferRequest

private[johnsnowlabs] class E5V(
    val onnxWrappers: Option[DecoderWrappers],
    val openvinoWrapper: Option[E5VWrappers],
    merges: Map[(String, String), Int],
    vocabulary: Map[String, Int],
    addedTokens: Map[String, Int],
    preprocessor: Preprocessor,
    generationConfig: GenerationConfig,
    imageToken: Int,
    imageGridPinpoints: Map[Int, Array[Int]],
    patchSize: Int)
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

  val bpeTokenizer: LLAVATokenizer = BpeTokenizer
    .forModel(
      "llava",
      merges = merges,
      vocab = vocabulary,
      specialTokens = Some(specialTokens),
      addPrefixSpaceToSentence = false,
      alwaysAddPrefix = false,
      prependString = "")
    .asInstanceOf[LLAVATokenizer]

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
  def encodeText(sentences: Seq[Annotation]): Seq[Array[Int]] = {

    val tokens = SentenceSplit
      .unpack(sentences)
      .map(s => {
        val sentWithTask = s
        bpeTokenizer
          .tokenize(sentWithTask)
          .map(bpeTokenizer.encode)
          .flatMap(_.map(_.pieceId))
      })
    tokens
  }

  def encode(
      imageAnnotations: Seq[AnnotationImage],
      sentences: Seq[Annotation],
      preprocessor: Preprocessor): (
      Seq[Array[Int]],
      Option[Array[Array[Array[Array[Array[Float]]]]]],
      Option[Array[(Int, Int)]]) = {
    val encodedText = encodeText(sentences).toArray

    // check if image annotations are present an height and width are > 0
    val imageAnnotationsFiltered =
      imageAnnotations.filter(annot => annot.width > 0 && annot.height > 0)

    val preprocessedImages = if (imageAnnotationsFiltered.nonEmpty) {
      Some(encodeImage(imageAnnotations.toArray, preprocessor))
    } else {
      None
    }
    val imageSizes = if (imageAnnotationsFiltered.nonEmpty) {
      Some(imageAnnotations.map(annot => (annot.width, annot.height)).toArray)
    } else {
      None
    }

    (encodedText, preprocessedImages, imageSizes)
  }

  def tag(
      batch: Seq[Array[Int]],
      images: Option[Array[Array[Array[Array[Array[Float]]]]]],
      imageSizes: Option[Array[(Int, Int)]]): Array[Array[Float]] = {

    val pixelValues = images
    val expandedDecoderInputsVals = batch
    val sequencesLength = expandedDecoderInputsVals.map(x => x.length).toArray
    val numReturn_sequences = 1
    // from config

    var effectiveBatch_size = 1
    var effectiveBatch_mult = 1

    effectiveBatch_size = expandedDecoderInputsVals.length
    effectiveBatch_mult = 1

    val inferRequestLanguageModel =
      openvinoWrapper.get.languageModel.getCompiledModel().create_infer_request()
    val inferRequestVisionEmbeddingsModel =
      openvinoWrapper.get.visionEmbeddingsModel.getCompiledModel().create_infer_request()
    val inferRequestTextEmbeddingsModel =
      openvinoWrapper.get.textEmbeddingsModel.getCompiledModel().create_infer_request()
    val inferRequestImagePackerModel =
      openvinoWrapper.get.imagePackerModel.getCompiledModel().create_infer_request()
    val inferRequestMergeModel =
      openvinoWrapper.get.mergeModel.getCompiledModel().create_infer_request()

    val generatedEmbeddings = getModelOutputs(
      decoderInputIds = expandedDecoderInputsVals.toArray,
      pixelValues = pixelValues,
      imageSizes = imageSizes,
      inferRequestLanguageModel = inferRequestLanguageModel,
      inferRequestVisionEmbeddingsModel = inferRequestVisionEmbeddingsModel,
      inferRequestTextEmbeddingsModel = inferRequestTextEmbeddingsModel,
      inferRequestImagePackerModel = inferRequestImagePackerModel,
      inferRequestMergeModel = inferRequestMergeModel)
    generatedEmbeddings
  }

  def predict(
      sentences: Seq[Annotation],
      imageAnnotations: Seq[AnnotationImage]): Seq[Annotation] = {

    val (encodedText, preprocessedImages, imageSizes) =
      encode(imageAnnotations, sentences, preprocessor)
    val sentenceEmbeddings = tag(encodedText, preprocessedImages, imageSizes)

    val annotations = sentences.zip(sentenceEmbeddings).map { case (sentence, vectors) =>
      Annotation(
        annotatorType = AnnotatorType.SENTENCE_EMBEDDINGS,
        begin = sentence.begin,
        end = sentence.end,
        result = sentence.result,
        metadata = sentence.metadata,
        embeddings = vectors)
    }
    annotations
  }

  def getModelOutputs(
      decoderInputIds: Array[Array[Int]],
      pixelValues: Option[Array[Array[Array[Array[Array[Float]]]]]],
      imageSizes: Option[Array[(Int, Int)]],
      inferRequestLanguageModel: InferRequest,
      inferRequestVisionEmbeddingsModel: InferRequest,
      inferRequestTextEmbeddingsModel: InferRequest,
      inferRequestImagePackerModel: InferRequest,
      inferRequestMergeModel: InferRequest): Array[Array[Float]] = {

    val (inputIdsLong, inputPositionIDsLong): (Array[Long], Array[Long]) = {
      // First pass
      val inpIdsLong = decoderInputIds.flatMap { tokenIds => tokenIds.map(_.toLong) }
      val posIdsLong = decoderInputIds.flatMap { tokenIds =>
        tokenIds.zipWithIndex.map { case (_, i) =>
          i.toLong
        }
      }
      (inpIdsLong, posIdsLong)
    }

    val attentionMask: Array[Long] = decoderInputIds.flatMap { tokenIds => tokenIds.map(_ => 1L) }
    val batchSize: Int = decoderInputIds.length
    val shape: Array[Int] = Array(batchSize, inputIdsLong.length / batchSize)

    val decoderAttentionMask: org.intel.openvino.Tensor =
      new org.intel.openvino.Tensor(Array(batchSize, decoderInputIds.head.length), attentionMask)
    val decoderPositionIDs: org.intel.openvino.Tensor =
      new org.intel.openvino.Tensor(shape, inputPositionIDsLong)

    val (finalEmbeds, finalAttentionMask, finalPositionIds) = getMultimodalEmbeddings(
      decoderInputIds,
      pixelValues,
      imageSizes,
      decoderAttentionMask,
      inferRequestVisionEmbeddingsModel,
      inferRequestTextEmbeddingsModel,
      inferRequestImagePackerModel,
      inferRequestMergeModel)

    inferRequestLanguageModel.set_tensor("inputs_embeds", finalEmbeds)
    if (finalAttentionMask.isDefined) {
      val finalAttentionMaskFloatTensor = new org.intel.openvino.Tensor(
        finalAttentionMask.get.get_shape(),
        // flat array of floats of values 1.0
        Array.fill(finalAttentionMask.get.get_shape().product)(1.0f))
      inferRequestLanguageModel.set_tensor("attention_mask", finalAttentionMaskFloatTensor)
    } else {
      val attentionMaskFloat: Array[Float] =
        decoderInputIds.flatMap { tokenIds => tokenIds.map(_ => 1f) }
      val attentionMaskFloatTensor =
        new org.intel.openvino.Tensor(
          Array(batchSize, decoderInputIds.head.length),
          attentionMaskFloat)
      inferRequestLanguageModel.set_tensor("attention_mask", attentionMaskFloatTensor)
    }
    if (finalPositionIds.isDefined) {
      inferRequestLanguageModel.set_tensor("position_ids", finalPositionIds.get)
    } else {
      inferRequestLanguageModel.set_tensor("position_ids", decoderPositionIDs)
    }
    inferRequestLanguageModel.infer()

    val result = inferRequestLanguageModel.get_tensor("last_hidden_state")
    val hiddenStateData = result.data()
    val hiddenStateShape = result.get_shape()
    val batchSizeResult = hiddenStateShape(0)
    val hiddenSize = hiddenStateShape(1)
    // Reshape to (batch, hidden_size) and return as Array[Array[Float]]
    Array.tabulate(batchSizeResult) { b =>
      val start = b * hiddenSize
      val end = start + hiddenSize
      hiddenStateData.slice(start, end)
    }

  }

  private def encodeImage(
      annotations: Array[AnnotationImage],
      preprocessor: Preprocessor): Array[Array[Array[Array[Array[Float]]]]] = {

    val batchProcessedImages = annotations.map { annot =>
      val bufferedImage = ImageIOUtils.byteToBufferedImage(
        bytes = annot.result,
        w = annot.width,
        h = annot.height,
        nChannels = annot.nChannels)
      val bestResolution = E5VUtils.selectBestResolution(
        (bufferedImage.getHeight, bufferedImage.getWidth),
        imageGridPinpoints.map { case (_, pinpoints) =>
          (pinpoints(0), pinpoints(1))
        }.toList)

      val (newHeight, newWidth) = E5VUtils.getPatchOutputSize(bufferedImage, bestResolution)
      val resizedForPatches = ImageResizeUtils.resizeBufferedImage(
        width = newWidth,
        height = newHeight,
        resample = preprocessor.resample)(bufferedImage)

      val paddedForPatches = E5VUtils.padImage(resizedForPatches, bestResolution)

      var patches = E5VUtils.divideToPatches(paddedForPatches, patchSize)

      // add the reshaped original image as the first patch
      val resizedOriginalImage = ImageResizeUtils.resizeBufferedImage(
        width = preprocessor.size,
        height = preprocessor.size,
        resample = preprocessor.resample)(bufferedImage)

      patches = List(resizedOriginalImage) ++ patches
      patches.map { patch =>
        ImageResizeUtils.normalizeAndConvertBufferedImage(
          img = patch,
          mean = preprocessor.image_mean,
          std = preprocessor.image_std,
          doNormalize = preprocessor.do_normalize,
          doRescale = preprocessor.do_rescale,
          rescaleFactor = preprocessor.rescale_factor)
      }.toArray
    }

    batchProcessedImages

  }

  def getMultimodalEmbeddings(
      inputIds: Array[Array[Int]],
      pixelValues: Option[Array[Array[Array[Array[Array[Float]]]]]],
      imageSizes: Option[Array[(Int, Int)]],
      attentionMask: org.intel.openvino.Tensor,
      inferRequestVisionEmbeddingsModel: InferRequest,
      inferRequestTextEmbeddingsModel: InferRequest,
      inferRequestImagePackerModel: InferRequest,
      inferRequestMergeModel: InferRequest): (
      org.intel.openvino.Tensor,
      Option[org.intel.openvino.Tensor],
      Option[org.intel.openvino.Tensor]) = {

    val inputIdsLong: Array[Long] = inputIds.flatMap(_.map(_.toLong))
    val batchSize: Int = inputIds.length
    val shape: Array[Int] = Array(batchSize, inputIdsLong.length / batchSize)
    val inputIdsLongTensor = new org.intel.openvino.Tensor(shape, inputIdsLong)

    // If pixelValues and imageSizes are present, do multimodal
    (pixelValues, imageSizes, attentionMask) match {
      case (Some(pixels), Some(sizes), attnMask) if pixels.nonEmpty && sizes.nonEmpty =>
        // 1. Get image features
        val pixelShape = Array(
          pixels.length,
          pixels.head.length,
          pixels.head.head.length,
          pixels.head.head.head.length,
          pixels.head.head.head.head.length)
        // Flatten the pixel values to match the expected input shape
        val flattenedPixels = pixels.flatten.flatten.flatten.flatten
        val pixelTensor =
          new org.intel.openvino.Tensor(pixelShape, flattenedPixels)

        inferRequestVisionEmbeddingsModel.set_tensor("pixel_values", pixelTensor)
        inferRequestVisionEmbeddingsModel.infer()
        val imageFeatures = inferRequestVisionEmbeddingsModel.get_output_tensor()

        // 2. Compute patch grid shape (dummy for now, should use config)
        val (numPatchHeight, numPatchWidth) =
          E5VUtils.getAnyResImageGridShape(
            imageSizes.get.head,
            imageGridPinpoints.map { case (_, pinpoints) =>
              (pinpoints(0), pinpoints(1))
            }.toList,
            preprocessor.size)

        // 3. Pack image features
        val imageSizesTensor = new org.intel.openvino.Tensor(
          Array(sizes.length, 2),
          sizes.flatMap(t => Array(t._1.toLong, t._2.toLong)))

        val numPatchHeightTensor =
          new org.intel.openvino.Tensor(Array[Int](), Array(numPatchHeight.toLong))

        val numPatchWidthTensor =
          new org.intel.openvino.Tensor(Array[Int](), Array(numPatchWidth.toLong))

        inferRequestImagePackerModel.set_tensor("image_feature", imageFeatures)
        inferRequestImagePackerModel.set_tensor("image_sizes", imageSizesTensor)
        inferRequestImagePackerModel.set_tensor("num_patch_height", numPatchHeightTensor)
        inferRequestImagePackerModel.set_tensor("num_patch_width", numPatchWidthTensor)
        inferRequestImagePackerModel.infer()

        val packedImageFeatures = inferRequestImagePackerModel.get_output_tensor()

        // 4. Get text embeddings
        inferRequestTextEmbeddingsModel.set_input_tensor(inputIdsLongTensor)
        inferRequestTextEmbeddingsModel.infer()
        val textEmbeddings = inferRequestTextEmbeddingsModel.get_output_tensor()

        // 5. Merge image and text embeddings
        inferRequestMergeModel.set_tensor("image_features", packedImageFeatures)
        inferRequestMergeModel.set_tensor("inputs_embeds", textEmbeddings)
        inferRequestMergeModel.set_tensor("input_ids", inputIdsLongTensor)

        inferRequestMergeModel.set_tensor("attention_mask", attnMask)
        inferRequestMergeModel.infer()
        (
          inferRequestMergeModel.get_tensor("final_embedding"),
          Some(inferRequestMergeModel.get_tensor("final_attention_mask")),
          Some(inferRequestMergeModel.get_tensor("position_ids")))
      case _ =>
        // Text-only
        inferRequestTextEmbeddingsModel.set_input_tensor(inputIdsLongTensor)
        inferRequestTextEmbeddingsModel.infer()
        (inferRequestTextEmbeddingsModel.get_output_tensor(), None, None)
    }
  }

}
