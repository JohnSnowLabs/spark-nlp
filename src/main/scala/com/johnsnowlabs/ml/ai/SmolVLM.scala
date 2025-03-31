package com.johnsnowlabs.ml.ai

import com.johnsnowlabs.ml.ai.util.Generation.GenerationConfig
import com.johnsnowlabs.ml.onnx.OnnxWrapper.DecoderWrappers
import com.johnsnowlabs.ml.openvino.OpenvinoWrapper.SmolVLMWrappers
import com.johnsnowlabs.nlp.annotators.common.Sentence
import com.johnsnowlabs.ml.util.{ONNX, Openvino}
import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common.SentenceSplit
import com.johnsnowlabs.nlp.annotators.cv.util.transform.{
  ImageResizeUtils,
  ImageSize,
  SmolVLMUtils
}
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

// {
//   "do_convert_rgb": true,
//   "do_image_splitting": true,
//   "do_normalize": true,
//   "do_pad": true,
//   "do_rescale": true,
//   "do_resize": true,
//   "image_mean": [
//     0.5,
//     0.5,
//     0.5
//   ],
//   "image_processor_type": "Idefics3ImageProcessor",
//   "image_std": [
//     0.5,
//     0.5,
//     0.5
//   ],
//   "max_image_size": {
//     "longest_edge": 512
//   },
//   "processor_class": "Idefics3Processor",
//   "resample": 1,
//   "rescale_factor": 0.00392156862745098,
//   "size": {
//     "longest_edge": 2048
//   }
// }

case class SmolVLMConfig(
    doResize: Boolean = true,
    size: Map[String, Int] = Map("longest_edge" -> 2048),
    maxImageSize: Map[String, Int] = Map("longest_edge" -> 512),
    doRescale: Boolean = true,
    rescaleFactor: Float = 1.0f / 255.0f,
    doNormalize: Boolean = true,
    imageMean: Array[Double] = Array(0.5, 0.5, 0.5),
    imageStd: Array[Double] = Array(0.5, 0.5, 0.5),
    doImageSplitting: Boolean = true,
    doPad: Boolean = true,
    resample: Int = 1,
    doConvertRgb: Boolean = true,
    returnPixelMask: Boolean = true)

private[johnsnowlabs] class SmolVLM(
    val onnxWrappers: Option[DecoderWrappers],
    val openvinoWrapper: Option[SmolVLMWrappers],
    merges: Map[(String, String), Int],
    vocabulary: Map[String, Int],
    addedTokens: Map[String, Int],
    preprocessor: Preprocessor,
    generationConfig: GenerationConfig,
    imageToken: Int,
    maxImageTiles: Int = 4,
    numVisionTokens: Int = 1601,
    paddingConstant: Int = 0,
    config: SmolVLMConfig = SmolVLMConfig())
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

  val bpeTokenizer: SmolVLMTokenizer = BpeTokenizer
    .forModel(
      "smolvlm",
      merges = merges,
      vocab = vocabulary,
      specialTokens = Some(specialTokens),
      addPrefixSpaceToSentence = false,
      alwaysAddPrefix = true,
      prependString = "")
    .asInstanceOf[SmolVLMTokenizer]

  def decode(sentences: Array[Array[Int]]): Seq[String] = {
    sentences.map(s => bpeTokenizer.decodeTokens(s.map(_.toInt)))
  }

  def encodeText(sentences: Seq[Annotation]): Seq[Array[Int]] = {
    val pattern = raw"<\|image\|>".r

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

  private def preprocessImage(
      image: BufferedImage,
      returnRowColInfo: Boolean = false): (SmolVLMUtils.BatchFeature, Option[(Int, Int)]) = {
    var processedImage = image
    var outputSplitResult: Option[SmolVLMUtils.SplitImageResult] = None
    // Resize if needed
    if (config.doResize) {
      processedImage = SmolVLMUtils.resizeWithLongestEdge(
        processedImage,
        longestEdge = config.size("longest_edge"),
        resample = config.resample)
    }

    // Handle image splitting
    if (config.doImageSplitting) {
      val resizedForEncoder = SmolVLMUtils.resizeForVisionEncoder(
        processedImage,
        config.maxImageSize("longest_edge"),
        config.resample)

      val splitResult: SmolVLMUtils.SplitImageResult = SmolVLMUtils.splitImage(
        resizedForEncoder,
        config.maxImageSize("longest_edge"),
        config.resample)

      processedImage = splitResult.frames.head // Use the first frame
      outputSplitResult = Some(splitResult)
    } else {
      // Square the images to maxImageSize
      processedImage = resizeBufferedImage(
        config.maxImageSize("longest_edge"),
        config.maxImageSize("longest_edge"),
        config.resample)(processedImage)
      outputSplitResult = Some(SmolVLMUtils.SplitImageResult(Seq(processedImage), 0, 0))
    }

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
    if (config.doPad) {
      val paddedImages: SmolVLMUtils.BatchFeature = SmolVLMUtils.pad(
        images = Seq(normalizedImages.toSeq),
        constantValue = paddingConstant,
        returnPixelMask = config.returnPixelMask)
    }
    val rowColInfo = outputSplitResult.map(r => (r.numSplitsH, r.numSplitsW))
    (paddedImages, rowColInfo)
  }

  private def encode(
      imageAnnotations: Seq[AnnotationImage],
      sentences: Seq[Annotation],
      preprocessor: Preprocessor): Map[String, Any] = {
    val processedImages = imageAnnotations.map { annot =>
      val bufferedImage = ImageIOUtils.byteToBufferedImage(
        bytes = annot.result,
        w = annot.width,
        h = annot.height,
        nChannels = annot.nChannels)
      preprocessImage(bufferedImage, returnRowColInfo = true)
    }

    val (pixelValuesWithMask, rowColInfo) = processedImages.unzip
    val pixelValues = pixelValuesWithMask.map(_.paddedImages)
    val pixelAttentionMasks = pixelValuesWithMask.map(_.pixelMasks)

    val encodedText = encodeText(sentences).toArray

    Map(
      "pixel_values" -> pixelValues.toArray,
      "pixel_attention_masks" -> pixelAttentionMasks.toArray,
      "encoded_text" -> encodedText,
      "rows" -> rowColInfo.map(_._1).toArray,
      "cols" -> rowColInfo.map(_._2).toArray)
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
    // TODO: Implement getModelOutputs and generation logic
    Seq.empty[Annotation]
  }
}
