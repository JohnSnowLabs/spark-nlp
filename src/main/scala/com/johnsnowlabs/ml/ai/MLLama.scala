package com.johnsnowlabs.ml.ai

import com.johnsnowlabs.ml.ai.util.Generation.GenerationConfig
import com.johnsnowlabs.ml.onnx.OnnxWrapper.DecoderWrappers
import com.johnsnowlabs.ml.openvino.OpenvinoWrapper.MLLamaWrappers
import com.johnsnowlabs.nlp.annotators.common.Sentence
import com.johnsnowlabs.ml.util.{ONNX, Openvino}
import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common.SentenceSplit
import com.johnsnowlabs.nlp.annotators.cv.util.transform.ImageResizeUtils
import com.johnsnowlabs.nlp.annotators.cv.util.transform.MllamaUtils

import com.johnsnowlabs.nlp.annotators.cv.feature_extractor.Preprocessor
import com.johnsnowlabs.nlp.annotators.cv.util.io.ImageIOUtils
import com.johnsnowlabs.nlp.annotators.tokenizer.bpe.{
  BpeTokenizer,
  MLLamaTokenizer,
  SpecialTokens
}
import org.intel.openvino.InferRequest

import scala.collection.JavaConverters._

private[johnsnowlabs] class MLLama(
    val onnxWrappers: Option[DecoderWrappers],
    val openvinoWrapper: Option[MLLamaWrappers],
    merges: Map[(String, String), Int],
    vocabulary: Map[String, Int],
    addedTokens: Map[String, Int],
    preprocessor: Preprocessor,
    generationConfig: GenerationConfig,
    imageToken: Int,
    maxImageTiles: Int = 4,
    numVisionTokens: Int = 1601,
    paddingConstant: Int = 0)
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

  val bpeTokenizer: MLLamaTokenizer = BpeTokenizer
    .forModel(
      "mllama",
      merges = merges,
      vocab = vocabulary,
      specialTokens = Some(specialTokens),
      addPrefixSpaceToSentence = false,
      alwaysAddPrefix = true,
      prependString = "")
    .asInstanceOf[MLLamaTokenizer]

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

    val pattern = raw"<\|image\|>".r

    // raise an error if the pattern is not found in the text
    if (pattern.findFirstIn(sentences.head.result).isEmpty) {
      throw new IllegalArgumentException("The pattern <\\|image\\|> is not found in the text")
    }

    val tokens = SentenceSplit
      .unpack(sentences)
      .map(s => {
        val sentWithTask = s
        Array(bosTokenId) ++ bpeTokenizer
          .tokenize(sentWithTask)
          .map(bpeTokenizer.encode)
          .flatMap(_.map(_.pieceId))
      })
    tokens
  }

  private def encode(
      imageAnnotations: Seq[AnnotationImage],
      sentences: Seq[Annotation],
      preprocessor: Preprocessor): Map[String, Any] = {
    val (preprocessedImages, aspectRatioIds, aspectRatioMask, numTiles) =
      encodeImage(imageAnnotations.toArray, preprocessor, maxImageTiles, paddingConstant)
    val encodedText = encodeText(sentences).toArray

    val crossAttentionMask = encodedText.map { sentence =>
      MllamaUtils.getCrossAttentionTokenMask(sentence, imageToken)
    }
    val maxLength = encodedText.map(_.length).max
    val crossAttentionMaskDense = MllamaUtils.convertSparseCrossAttentionMaskToDense(
      crossAttentionMask,
      numTiles.map(_.toArray).toArray,
      maxImageTiles,
      maxLength)

    Map(
      "pixelValues" -> preprocessedImages,
      "aspectRatioIds" -> aspectRatioIds,
      "aspectRatioMask" -> aspectRatioMask,
      "crossAttentionMask" -> crossAttentionMaskDense,
      "numTiles" -> numTiles,
      "encodedText" -> encodedText)

  }

  def tag(
      inputs: Map[String, Any],
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

    val inputIds = inputs("encodedText").asInstanceOf[Array[Array[Int]]]
    val ignoreTokenIdsInt = ignoreTokenIds
    val expandedDecoderInputsVals = inputIds
    val sequencesLength = expandedDecoderInputsVals.map(x => x.length).toArray
    val maxSentenceLength = sequencesLength.max // - curLen
    val numReturn_sequences = 1
    // from config

    var effectiveBatch_size = 1
    var effectiveBatch_mult = 1

    if (doSample) {
      effectiveBatch_size = expandedDecoderInputsVals.length * numReturn_sequences
      effectiveBatch_mult = numReturn_sequences
    } else {
      effectiveBatch_size = expandedDecoderInputsVals.length
      effectiveBatch_mult = 1
    }
    val inferRequestLanguageModel: InferRequest =
      openvinoWrapper.get.languageModel.getCompiledModel().create_infer_request()
    val inferRequestVisionEmbeddingsModel: InferRequest =
      openvinoWrapper.get.visionEmbeddingsModel.getCompiledModel().create_infer_request()
    val inferRequestReshapeModel: InferRequest =
      openvinoWrapper.get.reshapeModel.getCompiledModel().create_infer_request()
    val generatedIds = generateGreedy(
      inputIds,
      inputIds,
      inputs,
      maxOutputLength,
      inferRequestLanguageModel,
      inferRequestVisionEmbeddingsModel,
      inferRequestReshapeModel)
    generatedIds
  }

  def generateGreedy(
      encoderInputIds: Array[Array[Int]],
      decoderInputIds: Array[Array[Int]],
      inputs: Map[String, Any],
      maxOutputLength: Int,
      inferRequestLanguageModel: InferRequest,
      inferRequestVisionEmbeddingsModel: InferRequest,
      inferRequestReshapeModel: InferRequest): Array[Array[Int]] = {

    var generatedIds: Array[Array[Int]] = Array()
    var decoderInputIdsCopied = decoderInputIds.clone()
    val pixelValues =
      inputs("pixelValues").asInstanceOf[Array[Array[Array[Array[Array[Array[Float]]]]]]]
    val aspectRatioIds = inputs("aspectRatioIds").asInstanceOf[Array[Array[Int]]]
    val aspectRatioMask = inputs("aspectRatioMask").asInstanceOf[Array[Array[Array[Int]]]]

    val crossAttentionKeyValues = getCrossAttentionKeyValues(
      encoderInputIds,
      decoderInputIds,
      pixelValues,
      aspectRatioIds,
      aspectRatioMask,
      inferRequestVisionEmbeddingsModel)

    while (!greedyGenerationFinished(generatedIds, eosTokenId, maxOutputLength)) {
      val decoderOutputs = getModelOutputs(
        encoderInputIds,
        decoderInputIdsCopied,
        inputs,
        crossAttentionKeyValues,
        inferRequestLanguageModel)

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

    val inputs = encode(imageAnnotations, sentences, preprocessor)

    val tagged = tag(
      inputs,
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
      inputs: Map[String, Any],
      crossAttentionKeyValues: Array[(String, org.intel.openvino.Tensor)],
      inferRequestLanguageModel: InferRequest): Array[Array[Float]] = {
    val inferRequestReshapeModel =
      openvinoWrapper.get.reshapeModel.getCompiledModel().create_infer_request()

    val numTiles = inputs("numTiles").asInstanceOf[List[List[Int]]]
    val (inputIdsLong, inputPositionIDsLong, crossAttentionMaskDense)
        : (Array[Long], Array[Long], Array[Array[Array[Array[Int]]]]) =
      if (encoderInputIds.head.length == decoderInputIds.head.length) {
        // First pass
        val inpIdsLong = decoderInputIds.flatMap { tokenIds => tokenIds.map(_.toLong) }
        val posIdsLong = decoderInputIds.flatMap { tokenIds =>
          tokenIds.zipWithIndex.map { case (_, i) =>
            i.toLong
          }
        }
        val crossAttentionMask =
          inputs("crossAttentionMask").asInstanceOf[Array[Array[Array[Array[Int]]]]]
        (inpIdsLong, posIdsLong, crossAttentionMask)
      } else {
        // Subsequent passes
        val inpIdsLong = decoderInputIds.map { tokenIds => tokenIds.last.toLong }
        val posIdsLong = decoderInputIds.map { tokenIds =>
          tokenIds.zipWithIndex.map { case (_, i) =>
            i.toLong
          }.last
        }
        val crossAttentionMask = decoderInputIds.map { sentence =>
          MllamaUtils.getCrossAttentionTokenMask(sentence, imageToken)
        }
        val maxLength = decoderInputIds.map(_.length).max
        val crossAttentionMaskDense = MllamaUtils.convertSparseCrossAttentionMaskToDense(
          crossAttentionMask,
          numTiles.map(_.toArray).toArray,
          maxImageTiles,
          maxLength)
        (inpIdsLong, posIdsLong, crossAttentionMaskDense)
      }
    val attentionMask: Array[Long] =
      decoderInputIds.flatMap { tokenIds => tokenIds.map(_ => 1L) }

    val batchSize: Int = decoderInputIds.length
    val beamIdx: Array[Int] = new Array[Int](batchSize)
    val shape: Array[Int] = Array(batchSize, inputIdsLong.length / batchSize)

    val inputIdsTensor: org.intel.openvino.Tensor =
      new org.intel.openvino.Tensor(shape, inputIdsLong)

    val decoderAttentionMask: org.intel.openvino.Tensor =
      new org.intel.openvino.Tensor(Array(batchSize, decoderInputIds.head.length), attentionMask)
    val decoderPositionIDs: org.intel.openvino.Tensor =
      new org.intel.openvino.Tensor(shape, inputPositionIDsLong)
    val beamIdxTensor: org.intel.openvino.Tensor =
      new org.intel.openvino.Tensor(Array(batchSize), beamIdx)

    val crossAttentionMaskDenseTensor: org.intel.openvino.Tensor =
      new org.intel.openvino.Tensor(
        Array(
          batchSize,
          crossAttentionMaskDense.head.length,
          crossAttentionMaskDense.head.head.length,
          crossAttentionMaskDense.head.head.head.length),
        crossAttentionMaskDense.flatten.flatten.flatten.map(_.toLong))

    val numVisionTokensTensor: org.intel.openvino.Tensor =
      new org.intel.openvino.Tensor(Array[Int](), Array(numVisionTokens.toLong))

    val pastCrossAttentionKVLength: org.intel.openvino.Tensor =
      new org.intel.openvino.Tensor(
        Array[Int](),
        Array(
          crossAttentionKeyValues.head._2
            .get_shape()(crossAttentionKeyValues.head._2.get_shape().length - 2)
            .toLong))
    inferRequestReshapeModel.set_tensor("current_input_ids", inputIdsTensor)
    inferRequestReshapeModel.set_tensor("attention_mask", decoderAttentionMask)
    inferRequestReshapeModel.set_tensor("cross_attention_mask", crossAttentionMaskDenseTensor)
    inferRequestReshapeModel.set_tensor("num_vision_tokens", numVisionTokensTensor)
    inferRequestReshapeModel.set_tensor("past_cross_attn_kv_length", pastCrossAttentionKVLength)

    inferRequestReshapeModel.infer()
    val crossAttentionMaskReshaped =
      if (encoderInputIds.head.length == decoderInputIds.head.length) {
        inferRequestReshapeModel.get_tensor("cross_attention_mask_first_pass")
      } else {
        inferRequestReshapeModel.get_tensor("cross_attention_mask_second_pass")
      }
    val cachePosition = inferRequestReshapeModel.get_tensor("cache_position")
    val fullTextRowMaskedOutMask =
      inferRequestReshapeModel.get_tensor("full_text_row_masked_out_mask")

    // recreate the tensors by extracting the values from the reshaped tensors

    val clonedCrossAttentionMaskReshapedTensor: org.intel.openvino.Tensor =
      new org.intel.openvino.Tensor(
        crossAttentionMaskReshaped.get_shape(),
        crossAttentionMaskReshaped.data().map(_.toFloat))

    val clonedCachePositionTensor: org.intel.openvino.Tensor =
      new org.intel.openvino.Tensor(
        cachePosition.get_shape(),
        cachePosition.as_int().map(_.toLong))

    val clonedFullTextRowMaskedOutMaskTensor: org.intel.openvino.Tensor =
      new org.intel.openvino.Tensor(
        fullTextRowMaskedOutMask.get_shape(),
        fullTextRowMaskedOutMask.data().map(_.toFloat))

//    val crossAttentionMaskReshapedTensor: org.intel.openvino.Tensor =
//      new org.intel.openvino.Tensor(
//        crossAttentionMaskReshaped.get_shape(),
//        crossAttentionMaskReshaped.as_int().map(_.toFloat))

    inferRequestLanguageModel.set_tensor("input_ids", inputIdsTensor)
    inferRequestLanguageModel.set_tensor("attention_mask", decoderAttentionMask)
    inferRequestLanguageModel.set_tensor("position_ids", decoderPositionIDs)
    inferRequestLanguageModel.set_tensor("beam_idx", beamIdxTensor)
    inferRequestLanguageModel.set_tensor(
      "cross_attention_mask",
      clonedCrossAttentionMaskReshapedTensor)
    inferRequestLanguageModel.set_tensor("cache_position", clonedCachePositionTensor)
    inferRequestLanguageModel.set_tensor(
      "full_text_row_masked_out_mask",
      clonedFullTextRowMaskedOutMaskTensor)

    for ((name, tensor) <- crossAttentionKeyValues) {
      inferRequestLanguageModel.set_tensor(name, tensor)
    }

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

  private def argmax(scores: Array[Float]): Int = {
    // Validate that the array is not empty
    require(scores.nonEmpty, "Input array must not be empty")

    // Initialize variables to track the maximum score and its index
    var maxIndex = 0
    var maxValue = scores(0)

    // Iterate through the array to find the maximum value and its index
    for (i <- 1 until scores.length) {
      if (scores(i) > maxValue) {
        maxValue = scores(i)
        maxIndex = i
      }
    }

    maxIndex
  }

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
      preprocessor: Preprocessor,
      maxImageTiles: Int,
      paddingConstant: Int): (
      Array[Array[Array[Array[Array[Array[Float]]]]]],
      Array[Array[Int]],
      Array[Array[Array[Int]]],
      List[List[Int]]) = {

    val processed: Array[(Array[Array[Array[Array[Float]]]], List[(Int, Int)])] =
      annotations.map { annot =>
        val bufferedImage = ImageIOUtils.byteToBufferedImage(
          bytes = annot.result,
          w = annot.width,
          h = annot.height,
          nChannels = annot.nChannels)

        val (resizedImage, (numTilesHeight, numTilesWidth)) =
          if (preprocessor.do_resize) {
            MllamaUtils.resizeImage(
              width = preprocessor.size,
              height = preprocessor.size,
              resample = preprocessor.resample,
              maxImageTiles = maxImageTiles)(bufferedImage)
          } else (bufferedImage, (annot.height, annot.width))

        val paddedImage = MllamaUtils.pad(
          image = resizedImage,
          paddingConstant = paddingConstant,
          aspectRatio = (numTilesHeight, numTilesWidth),
          tileHeight = preprocessor.size,
          tileWidth = preprocessor.size)

        val imageTiles: Array[Array[Array[Array[Float]]]] = MllamaUtils.splitToTiles(
          image = paddedImage,
          numTilesHeight = numTilesHeight,
          numTilesWidth = numTilesWidth,
          mean = preprocessor.image_mean,
          std = preprocessor.image_std,
          doNormalize = preprocessor.do_normalize,
          doRescale = preprocessor.do_rescale,
          rescaleFactor = preprocessor.rescale_factor)

        val aspectRatioList: List[(Int, Int)] = List((numTilesHeight, numTilesWidth))

        (imageTiles, aspectRatioList)
      }

    val (batchProcessedImages, batchAspectRatios) = processed.unzip

    val (images, numTiles) =
      MllamaUtils.packImages(
        batchImages = List(batchProcessedImages),
        maxImageTiles = maxImageTiles)

    val aspectRatioIds: Array[Array[Int]] =
      MllamaUtils.convertAspectRatiosToIds(
        batchAspectRatios.toList,
        maxImageTiles = maxImageTiles)

    val aspectRatioMask: Array[Array[Array[Int]]] =
      MllamaUtils.buildAspectRatioMask(batchAspectRatios.toList, maxImageTiles = maxImageTiles)

    (images, aspectRatioIds, aspectRatioMask, numTiles)

  }

  def getCrossAttentionKeyValues(
      encoderInputIds: Array[Array[Int]],
      decoderInputIds: Array[Array[Int]],
      pixelValues: Array[Array[Array[Array[Array[Array[Float]]]]]],
      aspectRatioIds: Array[Array[Int]],
      aspectRatioMask: Array[Array[Array[Int]]],
      inferRequestVisionEmbeddingsModel: InferRequest)
      : Array[(String, org.intel.openvino.Tensor)] = {

    // filter out the cross attention output names only containing the word "cross_attn_key_values"
    val crossAttentionOutputNames =
      openvinoWrapper.get.visionEmbeddingsModel
        .getCompiledModel()
        .outputs()
        .asScala
        .filter(_.get_any_name().contains("cross_attn_key_values"))
        .map(_.get_any_name())
        .toArray

    val crossAttentionKeyValues: Array[(String, org.intel.openvino.Tensor)] = {
      if (encoderInputIds.head.length == decoderInputIds.head.length) {
        val pixelValuesShape = Array(
          pixelValues.length,
          pixelValues.head.length,
          pixelValues.head.head.length,
          pixelValues.head.head.head.length,
          pixelValues.head.head.head.head.length,
          pixelValues.head.head.head.head.head.length)
        val pixelValuesTensor: org.intel.openvino.Tensor =
          new org.intel.openvino.Tensor(
            pixelValuesShape,
            pixelValues.flatten.flatten.flatten.flatten.flatten)

        val aspectRatioIdsShape = Array(aspectRatioIds.length, aspectRatioIds.head.length)
        val aspectRatioIdsTensor: org.intel.openvino.Tensor =
          new org.intel.openvino.Tensor(aspectRatioIdsShape, aspectRatioIds.flatten.map(_.toLong))

        val aspectRatioMaskShape = Array(
          aspectRatioMask.length,
          aspectRatioMask.head.length,
          aspectRatioMask.head.head.length)

        val aspectRatioMaskTensor: org.intel.openvino.Tensor = new org.intel.openvino.Tensor(
          aspectRatioMaskShape,
          aspectRatioMask.flatten.flatten.map(_.toLong))

        // Get image embeddings
        inferRequestVisionEmbeddingsModel.set_tensor("pixel_values", pixelValuesTensor)
        inferRequestVisionEmbeddingsModel.set_tensor("aspect_ratio_ids", aspectRatioIdsTensor)
        inferRequestVisionEmbeddingsModel.set_tensor("aspect_ratio_mask", aspectRatioMaskTensor)

        inferRequestVisionEmbeddingsModel.infer()

        val crossAttentionKeyValues: Array[(String, org.intel.openvino.Tensor)] =
          crossAttentionOutputNames.map { outputName =>
            (outputName, inferRequestVisionEmbeddingsModel.get_tensor(outputName))
          }
        // return the cross attention output names and the key values
        crossAttentionKeyValues
      } else {
        // shouldn't be called
        throw new IllegalArgumentException("Should not be called for subsequent passes")
        Array()
      }
    }
    crossAttentionKeyValues
  }

}
