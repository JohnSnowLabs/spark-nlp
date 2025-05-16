package com.johnsnowlabs.ml.ai

import com.johnsnowlabs.ml.ai.util.Generation.GenerationConfig
import com.johnsnowlabs.ml.onnx.OnnxWrapper.DecoderWrappers
import com.johnsnowlabs.ml.openvino.OpenvinoWrapper.SmolVLMWrappers
import com.johnsnowlabs.nlp.annotators.common.Sentence
import com.johnsnowlabs.ml.util.{ONNX, Openvino}
import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common.SentenceSplit
import com.johnsnowlabs.nlp.annotators.cv.util.transform.{ImageResizeUtils, SmolVLMUtils}
import com.johnsnowlabs.nlp.annotators.cv.util.transform.SmolVLMUtils.ImageSize
import com.johnsnowlabs.nlp.annotators.cv.feature_extractor.Preprocessor
import com.johnsnowlabs.nlp.annotators.cv.util.io.ImageIOUtils
import com.johnsnowlabs.nlp.annotators.tokenizer.bpe.{
  BpeTokenizer,
  SmolVLMTokenizer,
  SpecialTokens
}
import org.intel.openvino.InferRequest
import java.awt.image.BufferedImage
import ImageResizeUtils.resizeBufferedImage

import scala.collection.JavaConverters._

/** Configuration class for SmolVLM model parameters
  * @param doResize
  *   Whether to resize input images
  * @param size
  *   Target size for image resizing
  * @param maxImageSize
  *   Maximum size for image processing
  * @param doRescale
  *   Whether to rescale pixel values
  * @param rescaleFactor
  *   Factor for pixel value rescaling
  * @param doNormalize
  *   Whether to normalize pixel values
  * @param imageMean
  *   Mean values for image normalization
  * @param imageStd
  *   Standard deviation values for image normalization
  * @param doImageSplitting
  *   Whether to split large images
  * @param doPad
  *   Whether to pad images
  * @param resample
  *   Resampling method for image resizing
  * @param doConvertRgb
  *   Whether to convert images to RGB
  * @param imageToken
  *   Special token for image placeholders
  * @param imageTokenId
  *   Token ID for image placeholders
  * @param endOfUtteranceToken
  *   Token indicating end of utterance
  * @param globalImageToken
  *   Token for global image context
  * @param fakeImageToken
  *   Token for image padding
  * @param imageSeqLen
  *   Length of image sequence
  * @param paddingConstant
  *   Value used for padding
  * @param patchSize
  *   Size of image patches for processing
  * @param returnPixelMask
  *   Whether to return pixel attention masks
  */
case class SmolVLMConfig(
    doResize: Boolean = true,
    size: Map[String, Int] = Map("longest_edge" -> 1536),
    maxImageSize: Map[String, Int] = Map("longest_edge" -> 384),
    doRescale: Boolean = true,
    rescaleFactor: Double = 1.0 / 255.0,
    doNormalize: Boolean = true,
    imageMean: Array[Double] = Array(0.5, 0.5, 0.5),
    imageStd: Array[Double] = Array(0.5, 0.5, 0.5),
    doImageSplitting: Boolean = true,
    doPad: Boolean = true,
    resample: Int = 1,
    doConvertRgb: Boolean = true,
    imageToken: String = "<image>",
    imageTokenId: Int = 49153,
    endOfUtteranceToken: String = "<end_of_utterance>",
    globalImageToken: String = "<global-img>",
    fakeImageToken: String = "<fake_token_around_image>",
    imageSeqLen: Int = 81,
    paddingConstant: Double = 0.0,
    unkTokenId: Int = 0,
    patchSize: Int = 14,
    returnPixelMask: Boolean = true)

/** SmolVLM (Small Vision Language Model) implementation for multimodal processing This class
  * handles the processing of both image and text inputs, combining them for vision-language tasks
  * using OpenVINO for efficient inference.
  *
  * @param onnxWrappers
  *   Optional ONNX model wrappers
  * @param openvinoWrapper
  *   Optional OpenVINO model wrappers
  * @param merges
  *   BPE tokenizer merges
  * @param vocabulary
  *   Tokenizer vocabulary
  * @param addedTokens
  *   Additional special tokens
  * @param preprocessor
  *   Image preprocessor
  * @param generationConfig
  *   Configuration for text generation
  * @param config
  *   SmolVLM specific configuration
  */
private[johnsnowlabs] class SmolVLM(
    val onnxWrappers: Option[DecoderWrappers],
    val openvinoWrapper: Option[SmolVLMWrappers],
    merges: Map[(String, String), Int],
    vocabulary: Map[String, Int],
    addedTokens: Map[String, Int],
    preprocessor: Preprocessor,
    generationConfig: GenerationConfig,
    config: SmolVLMConfig = SmolVLMConfig())
    extends Serializable {

  /** Detects the inference engine being used (ONNX or OpenVINO) */
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
    unkTokenString = reversedVocabulary(config.unkTokenId),
    maskTokenString = reversedVocabulary(eosTokenId),
    padTokenString = reversedVocabulary(paddingTokenId),
    additionalStrings = addedTokens.keys.toArray)

  val bpeTokenizer: SmolVLMTokenizer = BpeTokenizer
    .forModel(
      "smolvlm",
      merges = merges,
      vocab = vocabulary,
      specialTokens = Some(specialTokens),
      addPrefixSpaceToSentence = false,
      alwaysAddPrefix = false,
      prependString = "")
    .asInstanceOf[SmolVLMTokenizer]

  /** Decodes tokenized sequences back into text
    * @param sentences
    *   Array of tokenized sequences
    * @return
    *   Decoded text sequences
    */
  def decode(sentences: Array[Array[Int]]): Seq[String] = {
    // Convert each sequence of token IDs back to text using the BPE tokenizer
    sentences.map(s => bpeTokenizer.decodeTokens(s.map(_.toInt)))
  }

  /** Encodes text sequences into token IDs
    * @param sentences
    *   Text sequences to encode
    * @return
    *   Encoded token sequences
    * @throws IllegalArgumentException
    *   if image token is not found in text
    */
  def encodeText(sentences: Seq[Annotation]): Seq[Array[Int]] = {
    // Check for presence of image token in text
    val pattern = raw"<image>".r
    if (pattern.findFirstIn(sentences.head.result).isEmpty) {
      throw new IllegalArgumentException("The pattern <\\image\\> is not found in the text")
    }

    // Process each sentence through the tokenizer pipeline
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

    // inject the image padding tokens of length imgTokenLen between the prompt chunks and reduce the Seq[Array[Array[Int]]] to Seq[Array[Int]]
    val tokens = promptChunks
      .zip(List(config.imageSeqLen))
      .map(s => {
        val (promptChunk, imgTokenLen) = s
        val imgPaddingTokens = Array.fill(imgTokenLen)(config.imageTokenId)
        val combinedChunks = promptChunk
          .map(_.toArray)
          .reduce(_ ++ imgPaddingTokens ++ _)
        Array(bosTokenId) ++ combinedChunks
      })
    tokens
  }

  /** Preprocesses input images according to model requirements
    * @param image
    *   Input image to preprocess
    * @param returnRowColInfo
    *   Whether to return row/column information
    * @return
    *   Tuple of processed image features and optional row/column info
    */
  private def preprocessImage(
      image: BufferedImage): (SmolVLMUtils.BatchFeature, Option[(Int, Int)]) = {
    var processedImage = image
    var outputSplitResult: Option[SmolVLMUtils.SplitImageResult] = None

    // Step 1: Resize image if configured
    if (config.doResize) {
      processedImage = SmolVLMUtils.resizeWithLongestEdge(
        processedImage,
        longestEdge = config.size("longest_edge"),
        resample = config.resample)
    }

    // Step 2: Handle image splitting based on configuration
    if (config.doImageSplitting) {
      // Resize image for vision encoder
      val resizedForEncoder = SmolVLMUtils.resizeForVisionEncoder(
        processedImage,
        config.maxImageSize("longest_edge"),
        config.resample)

      // Split image into tiles
      val splitResult: SmolVLMUtils.SplitImageResult = SmolVLMUtils.splitImage(
        resizedForEncoder,
        config.maxImageSize("longest_edge"),
        config.resample)

      outputSplitResult = Some(splitResult)
    } else {
      // Square the images to maxImageSize if not splitting
      processedImage = resizeBufferedImage(
        config.maxImageSize("longest_edge"),
        config.maxImageSize("longest_edge"),
        config.resample)(processedImage)
      outputSplitResult = Some(SmolVLMUtils.SplitImageResult(Seq(processedImage), 0, 0))
    }

    // Step 3: Normalize and convert images
    val normalizedImages =
      outputSplitResult.get.frames
        .map(frame =>
          ImageResizeUtils.normalizeAndConvertBufferedImage(
            img = frame,
            mean = config.imageMean,
            std = config.imageStd,
            doNormalize = config.doNormalize,
            doRescale = config.doRescale,
            rescaleFactor = config.rescaleFactor))

    // print image size
    val shape = Array(
      normalizedImages.length,
      normalizedImages.head.length,
      normalizedImages.head.head.length,
      normalizedImages.head.head.head.length)
    // Step 4: Pad images if configured
    var paddedImages: SmolVLMUtils.BatchFeature = null

    if (config.doPad) {

      paddedImages = SmolVLMUtils.pad(
        images = Seq(normalizedImages.toSeq),
        constantValue = config.paddingConstant.toFloat,
        returnPixelMask = config.returnPixelMask)
    } else {
      paddedImages =
        SmolVLMUtils.BatchFeature(paddedImages = Seq(normalizedImages.toSeq), pixelMasks = None)
    }

    // Extract row/column information if requested
    val rowColInfo = outputSplitResult.map(r => (r.numSplitsH, r.numSplitsW))
    (paddedImages, rowColInfo)
  }

  /** Encodes input data into model-compatible format
    * @param imageAnnotations
    *   Image annotations
    * @param sentences
    *   Text annotations
    * @param preprocessor
    *   Image preprocessor
    * @return
    *   Map of encoded inputs
    */
  private def encode(
      imageAnnotations: Seq[AnnotationImage],
      sentences: Seq[Annotation],
      preprocessor: Preprocessor): Map[String, Any] = {
    // Step 1: Process each image annotation
    val processedImages = imageAnnotations.map { annot =>
      // Convert bytes to BufferedImage
      val bufferedImage = ImageIOUtils.byteToBufferedImage(
        bytes = annot.result,
        w = annot.width,
        h = annot.height,
        nChannels = annot.nChannels)
      preprocessImage(bufferedImage)
    }

    // Step 2: Extract pixel values and masks
    val (pixelValuesWithMask, rowColInfo) = processedImages.unzip
    // collect pixel values to 5d array (batch, num_images, num_channels, height, width)
    // concatenate all the images in the batch
    var pixelValues = Seq[Array[Array[Array[Array[Float]]]]]()
    for (i <- pixelValuesWithMask) {
      pixelValues = pixelValues :+ i.paddedImages.head.toArray
    }

    val pixelAttentionMasks = pixelValuesWithMask.map(_.pixelMasks.head.toArray)

    // Step 3: Extract image grid information
    val imageRows: Array[Array[Int]] = Array(
      rowColInfo.map(_.getOrElse((0, 0))).map(_._1).toArray)
    val imageCols: Array[Array[Int]] = Array(
      rowColInfo.map(_.getOrElse((0, 0))).map(_._2).toArray)

    // Step 4: Process sentences
    val sentenceSplit = SentenceSplit
      .unpack(sentences)
      .map(_.content)

    // Step 5: Generate prompt strings with image tokens
    val promptStrings =
      sentenceSplit.zip(imageRows.zip(imageCols)).map { case (sample, (sampleRows, sampleCols)) =>
        // Generate image prompt strings for each grid position
        val imagePromptStrings = sampleRows.zip(sampleCols).map { case (nRows, nCols) =>
          SmolVLMUtils.getImagePromptString(
            nRows,
            nCols,
            config.imageSeqLen,
            imageToken = config.imageToken,
            fakeTokenAroundImage = config.fakeImageToken,
            globalImageToken = config.globalImageToken)
        }

        // Split sample by image token and combine with prompt strings
        val splitSample = sample.split(this.config.imageToken)
        if (splitSample.isEmpty) {
          throw new IllegalArgumentException("The image token should be present in the text.")
        }

        splitSample
          .zipAll(imagePromptStrings, "", "")
          .map { case (textPart, imagePromptString) =>
            textPart + imagePromptString
          }
          .mkString
      }

    // Step 6: Encode final prompt strings
    val promptStringAnnotations = promptStrings.map(s => Annotation(s))
    val encodedText = encodeText(promptStringAnnotations).toArray

    // Step 7: Return all processed data
    Map(
      "pixel_values" -> pixelValues.toArray,
      "pixel_attention_masks" -> pixelAttentionMasks.toArray,
      "encoded_text" -> encodedText,
      "rows" -> imageRows,
      "cols" -> imageCols,
      "prompt_strings" -> promptStrings)
  }

  /** Generates text output based on input
    * @param inputs
    *   Input data map
    * @param minOutputLength
    *   Minimum length of generated text
    * @param maxOutputLength
    *   Maximum length of generated text
    * @param doSample
    *   Whether to use sampling for generation
    * @param temperature
    *   Temperature for sampling
    * @param topK
    *   Top-k sampling parameter
    * @param topP
    *   Top-p sampling parameter
    * @param repetitionPenalty
    *   Penalty for repetition
    * @param noRepeatNgramSize
    *   Size of n-grams to avoid repeating
    * @param randomSeed
    *   Optional random seed
    * @param ignoreTokenIds
    *   Tokens to ignore during generation
    * @param beamSize
    *   Size of beam search
    * @param maxInputLength
    *   Maximum input length
    * @return
    *   Generated token sequences
    */
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
      randomSeed: Option[Long] = None,
      ignoreTokenIds: Array[Int] = Array(),
      beamSize: Int,
      maxInputLength: Int): Array[Array[Int]] = {
    // Step 1: Extract and validate input data
    val inputIds = inputs("encoded_text").asInstanceOf[Array[Array[Int]]]
    val ignoreTokenIdsInt = ignoreTokenIds
    val expandedDecoderInputsVals = inputIds
    val sequencesLength = expandedDecoderInputsVals.map(x => x.length).toArray
    val maxSentenceLength = sequencesLength.max
    val numReturn_sequences = 1

    // Step 2: Calculate batch sizes based on sampling mode
    var effectiveBatch_size = 1
    var effectiveBatch_mult = 1
    if (doSample) {
      effectiveBatch_size = beamSize
      effectiveBatch_mult = numReturn_sequences
    } else {
      effectiveBatch_size = expandedDecoderInputsVals.length
      effectiveBatch_mult = 1
    }

    // Step 3: Initialize inference requests for all models
    val inferRequestLanguageModel: InferRequest =
      openvinoWrapper.get.languageModel.getCompiledModel().create_infer_request()
    val inferRequestImageEmbedModel: InferRequest =
      openvinoWrapper.get.imageEmbedModel.getCompiledModel().create_infer_request()
    val inferRequestImageEncoderModel: InferRequest =
      openvinoWrapper.get.imageEncoderModel.getCompiledModel().create_infer_request()
    val inferRequestImageConnectorModel: InferRequest =
      openvinoWrapper.get.imageConnectorModel.getCompiledModel().create_infer_request()
    val inferRequestModelMergerModel: InferRequest =
      openvinoWrapper.get.modelMergerModel.getCompiledModel().create_infer_request()
    val inferRequestTextEmbeddingsModel: InferRequest =
      openvinoWrapper.get.textEmbeddingsModel.getCompiledModel().create_infer_request()
    val inferRequestLmHeadModel: InferRequest =
      openvinoWrapper.get.lmHeadModel.getCompiledModel().create_infer_request()

    // Step 4: Process pixel values
    val pixelValues =
      inputs("pixel_values").asInstanceOf[Array[Array[Array[Array[Array[Float]]]]]]
    val pixelValuesTensor = new org.intel.openvino.Tensor(
      Array(
        pixelValues.length,
        pixelValues.head.length,
        pixelValues.head.head.length,
        pixelValues.head.head.head.length,
        pixelValues.head.head.head.head.length),
      pixelValues.flatten.flatten.flatten.flatten.toArray)

    // Step 5: Get image embeddings
    val imageHiddenStates = getImageEmbeddings(
      pixelValuesTensor,
      inferRequestImageEmbedModel,
      inferRequestImageEncoderModel,
      inferRequestImageConnectorModel)

    // Step 6: Generate text using greedy decoding
    val generatedIds = generateGreedy(
      encoderInputIds = inputIds,
      decoderInputIds = inputIds,
      imageHiddenStates = imageHiddenStates,
      maxOutputLength = maxOutputLength,
      inferRequestModelMergerModel = inferRequestModelMergerModel,
      inferRequestTextEmbeddingsModel = inferRequestTextEmbeddingsModel,
      inferRequestLanguageModel = inferRequestLanguageModel,
      inferRequestLmHeadModel = inferRequestLmHeadModel)
    generatedIds
  }

  /** Checks if greedy generation is finished
    * @param decoderIds
    *   Current decoder output IDs
    * @param eosTokenId
    *   End of sequence token ID
    * @param maxOutputLength
    *   Maximum output length
    * @return
    *   Whether generation is complete
    */
  private def greedyGenerationFinished(
      decoderIds: Seq[Array[Int]],
      eosTokenId: Int,
      maxOutputLength: Int): Boolean = {
    // Check if generation is complete based on length or EOS token
    if (decoderIds.isEmpty) {
      false
    } else {
      decoderIds.forall { ids =>
        ids.length >= maxOutputLength || ids.last == eosTokenId
      }
    }
  }

  /** Finds index of maximum value in array
    * @param scores
    *   Array of scores
    * @return
    *   Index of maximum value
    * @throws IllegalArgumentException
    *   if input array is empty
    */
  private def argmax(scores: Array[Float]): Int = {
    // Validate input array
    require(scores.nonEmpty, "Input array must not be empty")

    // Initialize tracking variables
    var maxIndex = 0
    var maxValue = scores(0)

    // Find maximum value and its index
    for (i <- 1 until scores.length) {
      if (scores(i) > maxValue) {
        maxValue = scores(i)
        maxIndex = i
      }
    }

    maxIndex
  }

  /** Generates text using greedy decoding
    * @param encoderInputIds
    *   Encoder input token IDs
    * @param decoderInputIds
    *   Decoder input token IDs
    * @param imageHiddenStates
    *   Image embeddings
    * @param maxOutputLength
    *   Maximum output length
    * @param inferRequestModelMergerModel
    *   Model merger inference request
    * @param inferRequestTextEmbeddingsModel
    *   Text embeddings inference request
    * @param inferRequestLanguageModel
    *   Language model inference request
    * @param inferRequestLmHeadModel
    *   Language model head inference request
    * @return
    *   Generated token sequences
    */
  def generateGreedy(
      encoderInputIds: Array[Array[Int]],
      decoderInputIds: Array[Array[Int]],
      imageHiddenStates: org.intel.openvino.Tensor,
      maxOutputLength: Int,
      inferRequestModelMergerModel: InferRequest,
      inferRequestTextEmbeddingsModel: InferRequest,
      inferRequestLanguageModel: InferRequest,
      inferRequestLmHeadModel: InferRequest): Array[Array[Int]] = {
    // Initialize generation variables
    var generatedIds: Array[Array[Int]] = Array()
    var decoderInputIdsCopied = decoderInputIds.clone()

    // Generate tokens until completion criteria are met
    while (!greedyGenerationFinished(generatedIds, eosTokenId, maxOutputLength)) {
      // Get model outputs for current state
      val decoderOutputs = getModelOutputs(
        encoderInputIds,
        decoderInputIdsCopied,
        imageHiddenStates,
        inferRequestModelMergerModel,
        inferRequestTextEmbeddingsModel,
        inferRequestLanguageModel,
        inferRequestLmHeadModel)

      // Select next tokens using argmax
      val nextTokenIds = decoderOutputs.map { scores =>
        argmax(scores)
      }

      // Update generated sequences
      if (generatedIds.isEmpty) {
        generatedIds = nextTokenIds.map(Array(_))
      } else {
        generatedIds =
          generatedIds.zip(nextTokenIds).map { case (currentIds: Array[Int], nextId: Int) =>
            currentIds ++ Array(nextId)
          }
      }

      // Update decoder input for next iteration
      decoderInputIdsCopied =
        decoderInputIdsCopied.zip(nextTokenIds).map { case (currentIds, nextId) =>
          currentIds ++ Array(nextId)
        }
    }
    generatedIds
  }

  /** Processes images to generate embeddings
    * @param pixelValues
    *   Input pixel values tensor
    * @param inferRequestImageEmbed
    *   Image embedding inference request
    * @param inferRequestImageEncoder
    *   Image encoder inference request
    * @param inferRequestImageConnector
    *   Image connector inference request
    * @return
    *   Processed image embeddings
    */
  def getImageEmbeddings(
      pixelValues: org.intel.openvino.Tensor,
      inferRequestImageEmbed: InferRequest,
      inferRequestImageEncoder: InferRequest,
      inferRequestImageConnector: InferRequest): org.intel.openvino.Tensor = {
    // Step 1: Extract tensor dimensions
    val pixelValuesShape = pixelValues.get_shape()
    val batchSize = pixelValuesShape(0)
    val numImages = pixelValuesShape(1)
    val numChannels = pixelValuesShape(2)
    val height = pixelValuesShape(3)
    val width = pixelValuesShape(4)

    // Step 2: Reshape pixel values for processing
    val reshapedPixelValues = new org.intel.openvino.Tensor(
      Array(batchSize * numImages, numChannels, height, width),
      pixelValues.data())

    // Step 3: Calculate values per image
    val nbValuesPerImage = numChannels * height * width

    // Step 4: Create mask for real (non-zero) images
    val realImagesMask = reshapedPixelValues
      .data()
      .grouped(nbValuesPerImage)
      .map { values =>
        values.exists(_ != 0.0f)
      }
      .toArray

    // Step 5: Handle case with no real images
    if (!realImagesMask.exists(identity)) {
      realImagesMask(0) = true
    }

    // Step 6: Filter to keep only real images
    val realPixelValues = reshapedPixelValues
      .data()
      .zipWithIndex
      .filter { case (_, idx) => realImagesMask(idx / nbValuesPerImage) }
      .map(_._1)

    // Step 7: Create patch attention mask
    val patchSize = config.patchSize
    val numPatchesH = height / patchSize
    val numPatchesW = width / patchSize
    val patchAttentionMask = new org.intel.openvino.Tensor(
      Array(realPixelValues.length / (numChannels * height * width), numPatchesH, numPatchesW),
      Array.fill(
        realPixelValues.length / (numChannels * height * width) * numPatchesH * numPatchesW)(1L))

    // Step 8: Create tensor for real pixel values
    val pixelValuesTensor = new org.intel.openvino.Tensor(
      Array(realPixelValues.length / (numChannels * height * width), numChannels, height, width),
      pixelValues.data())

    // Step 9: Get initial image embeddings
    inferRequestImageEmbed.set_tensor("pixel_values", pixelValuesTensor)
    inferRequestImageEmbed.set_tensor("patch_attention_mask", patchAttentionMask)
    inferRequestImageEmbed.infer()
    val hiddenStates = inferRequestImageEmbed.get_output_tensor()

    // Step 10: Process through image encoder
    inferRequestImageEncoder.set_tensor("inputs_embeds", hiddenStates)
    inferRequestImageEncoder.infer()
    val imageHiddenStatesBefore = inferRequestImageEncoder.get_output_tensor()

    // Step 11: Process through image connector
    inferRequestImageConnector.set_tensor("image_hidden_states", imageHiddenStatesBefore)
    inferRequestImageConnector.infer()
    val imageHiddenStates = inferRequestImageConnector.get_output_tensor()

    imageHiddenStates
  }

  /** Generates predictions for input data
    * @param sentences
    *   Input text sequences
    * @param imageAnnotations
    *   Input image annotations
    * @param batchSize
    *   Size of processing batch
    * @param minOutputLength
    *   Minimum output length
    * @param maxOutputLength
    *   Maximum output length
    * @param doSample
    *   Whether to use sampling
    * @param temperature
    *   Temperature for sampling
    * @param topK
    *   Top-k sampling parameter
    * @param topP
    *   Top-p sampling parameter
    * @param repetitionPenalty
    *   Penalty for repetition
    * @param noRepeatNgramSize
    *   Size of n-grams to avoid repeating
    * @param randomSeed
    *   Optional random seed
    * @param ignoreTokenIds
    *   Tokens to ignore
    * @param beamSize
    *   Size of beam search
    * @param maxInputLength
    *   Maximum input length
    * @return
    *   Generated text annotations
    */
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
    // Step 1: Encode input data
    val inputs = encode(imageAnnotations, sentences, preprocessor)

    // Step 2: Generate text using tag method
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
      maxInputLength)

    // Step 3: Decode generated tokens to text
    val decoded = decode(tagged)

    // Step 4: Create annotations from decoded text
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

  /** Gets model outputs for given inputs
    * @param encoderInputIds
    *   Encoder input token IDs
    * @param decoderInputIds
    *   Decoder input token IDs
    * @param imageEmbeddings
    *   Image embeddings
    * @param inferRequestModelMergerModel
    *   Model merger inference request
    * @param inferRequestTextEmbeddingsModel
    *   Text embeddings inference request
    * @param inferRequestLanguageModel
    *   Language model inference request
    * @param inferRequestLmHeadModel
    *   Language model head inference request
    * @return
    *   Model output logits
    */
  def getModelOutputs(
      encoderInputIds: Array[Array[Int]],
      decoderInputIds: Array[Array[Int]],
      imageEmbeddings: org.intel.openvino.Tensor,
      inferRequestModelMergerModel: InferRequest,
      inferRequestTextEmbeddingsModel: InferRequest,
      inferRequestLanguageModel: InferRequest,
      inferRequestLmHeadModel: InferRequest): Array[Array[Float]] = {
    // Step 1: Get batch size
    val batchSize: Int = decoderInputIds.length

    // Step 2: Process input IDs and position IDs
    val (inputIdsLong, inputPositionIDsLong): (Array[Long], Array[Long]) =
      if (encoderInputIds.head.length == decoderInputIds.head.length) {
        // First pass: process all tokens
        val inpIdsLong = decoderInputIds.flatMap { tokenIds => tokenIds.map(_.toLong) }
        val posIdsLong = decoderInputIds.flatMap { tokenIds =>
          tokenIds.zipWithIndex.map { case (_, i) =>
            i.toLong
          }
        }
        (inpIdsLong, posIdsLong)
      } else {
        // Subsequent passes: process only last token
        val inpIdsLong = decoderInputIds.map { tokenIds => tokenIds.last.toLong }
        val posIdsLong = decoderInputIds.map { tokenIds =>
          tokenIds.zipWithIndex.map { case (_, i) =>
            i.toLong
          }.last
        }
        (inpIdsLong, posIdsLong)
      }

    // Step 3: Create input tensor
    val inputIdsTensor = new org.intel.openvino.Tensor(
      Array(batchSize, inputIdsLong.length / batchSize),
      inputIdsLong)

    // Step 4: Get text embeddings
    inferRequestTextEmbeddingsModel.set_input_tensor(inputIdsTensor)
    inferRequestTextEmbeddingsModel.infer()
    val textEmbeddings = inferRequestTextEmbeddingsModel.get_output_tensor()

    // Step 5: Merge text and image embeddings
    inferRequestModelMergerModel.set_tensor("input_ids", inputIdsTensor)
    inferRequestModelMergerModel.set_tensor("inputs_embeds", textEmbeddings)
    inferRequestModelMergerModel.set_tensor("image_hidden_states", imageEmbeddings)
    inferRequestModelMergerModel.infer()
    val mergedEmbeddings = inferRequestModelMergerModel.get_output_tensor()

    // Step 6: Create attention masks and position IDs
    val attentionMask: Array[Long] =
      decoderInputIds.flatMap { tokenIds => tokenIds.map(_ => 1L) }
    val beamIdx: Array[Int] = new Array[Int](batchSize)
    val shape: Array[Int] = Array(batchSize, inputIdsLong.length / batchSize)

    // Step 7: Create tensors for attention and position information
    val decoderAttentionMask: org.intel.openvino.Tensor =
      new org.intel.openvino.Tensor(Array(batchSize, decoderInputIds.head.length), attentionMask)
    val decoderPositionIDs: org.intel.openvino.Tensor =
      new org.intel.openvino.Tensor(shape, inputPositionIDsLong)
    val beamIdxTensor: org.intel.openvino.Tensor =
      new org.intel.openvino.Tensor(Array(batchSize), beamIdx)

    // Step 8: Create tensor for merged embeddings
    val imgEmbeddingTensor =
      new org.intel.openvino.Tensor(mergedEmbeddings.get_shape(), mergedEmbeddings.data())

    // Step 9: Run language model inference
    inferRequestLanguageModel.set_tensor("inputs_embeds", imgEmbeddingTensor)
    inferRequestLanguageModel.set_tensor("attention_mask", decoderAttentionMask)
    inferRequestLanguageModel.set_tensor("position_ids", decoderPositionIDs)
    inferRequestLanguageModel.set_tensor("beam_idx", beamIdxTensor)
    inferRequestLanguageModel.infer()

    // Step 10: Get last hidden state
    val result = inferRequestLanguageModel.get_tensor("last_hidden_state")

    // Step 11: Run language model head inference
    inferRequestLmHeadModel.set_input_tensor(result)
    inferRequestLmHeadModel.infer()
    val logit = inferRequestLmHeadModel.get_output_tensor()
    val logitRaw: Array[Float] = logit.data()
    // Step 12: Process logits to get output scores
    val sequenceLength = inputIdsLong.length / batchSize
    val decoderOutputs = (0 until batchSize).map(i => {
      logitRaw
        .slice(
          i * sequenceLength * vocabSize + (sequenceLength - 1) * vocabSize,
          i * sequenceLength * vocabSize + sequenceLength * vocabSize)
    })
    decoderOutputs.toArray
  }

}
