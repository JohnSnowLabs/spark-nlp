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

import breeze.optimize.BatchSize
import com.johnsnowlabs.ml.ai.util.Generation.GenerationConfig
import com.johnsnowlabs.ml.onnx.OnnxWrapper.DecoderWrappers
import com.johnsnowlabs.ml.openvino.OpenvinoWrapper.Qwen2VLWrappers
import com.johnsnowlabs.nlp.annotators.common.Sentence
import com.johnsnowlabs.ml.util.{ONNX, Openvino}
import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common.SentenceSplit
import com.johnsnowlabs.nlp.annotators.cv.feature_extractor.Preprocessor
import com.johnsnowlabs.nlp.annotators.cv.util.io.ImageIOUtils
import com.johnsnowlabs.nlp.annotators.cv.util.transform.ImageResizeUtils
import com.johnsnowlabs.nlp.annotators.cv.util.transform.Qwen2VLUtils.{
  IMAGE_FACTOR,
  MAX_PIXELS,
  MAX_RATIO,
  MIN_PIXELS,
  imageBufferToArray,
  smartResize
}
import com.johnsnowlabs.nlp.annotators.tokenizer.bpe.{
  BpeTokenizer,
  LLAMA3Tokenizer,
  Qwen2VLTokenizer,
  SpecialTokens
}
import org.intel.openvino.InferRequest

import scala.collection.JavaConverters._

private[johnsnowlabs] class Qwen2VL(
    val onnxWrappers: Option[DecoderWrappers],
    val openvinoWrapper: Option[Qwen2VLWrappers],
    merges: Map[(String, String), Int],
    vocabulary: Map[String, Int],
    addedTokens: Map[String, Int],
    preprocessor: Preprocessor,
    generationConfig: GenerationConfig,
    minPixels: Int = MIN_PIXELS,
    maxPixels: Int = MAX_PIXELS,
    imageToken: Int = 151655)
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

  val bpeTokenizer: Qwen2VLTokenizer = BpeTokenizer
    .forModel(
      "qwen2vl",
      merges = merges,
      vocab = vocabulary,
      specialTokens = Some(specialTokens),
      addPrefixSpaceToSentence = false,
      alwaysAddPrefix = false,
      prependString = "")
    .asInstanceOf[Qwen2VLTokenizer]

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

//    val pattern = raw"<\|image_\d+\|>".r
//    <|vision_start|><|image_pad|><|vision_end|>

    val pattern = raw"<\|image_pad\|>".r
    // raise an error if the pattern is not found in the text
    if (pattern.findFirstIn(sentences.head.result).isEmpty) {
      throw new IllegalArgumentException("The pattern <\\|image_pad\\|> is not found in the text")
    }

    // split the sentences into chunks based on the pattern and tokenize them
    // eg in python prompt_chunks = [self.tokenizer(chunk).input_ids for chunk in re.split(pattern, texts)]
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
      .zip(imgTokenLen)
      .map(s => {
        val (promptChunk, imgTokenLen) = s
        val imgPaddingTokens = Array.fill(imgTokenLen)(imageToken)
        val combinedChunks = promptChunk
          .map(_.toArray)
          .reduce(_ ++ imgPaddingTokens ++ _)
        Array(bosTokenId) ++ combinedChunks
      })

    //    val tokens = SentenceSplit
    //      .unpack(sentences)
    //      .map(s => {
    //        val sentWithTask = s
    //        bpeTokenizer
    //          .tokenize(sentWithTask)
    //          .map(bpeTokenizer.encode)
    //          .flatMap(_.map(_.pieceId))
    //      })
    tokens
  }
  def encode(
      imageAnnotations: Seq[AnnotationImage],
      sentences: Seq[Annotation],
      preprocessor: Preprocessor)
      : (Seq[Array[Int]], (org.intel.openvino.Tensor, (Int, Int, Int))) = {
    val preprocessedImages = preprocessImage(
      imageAnnotations,
      preprocessor,
      minPixels = minPixels,
      maxPixels = maxPixels)
    val imageTokenLength = preprocessedImages._2._2 * preprocessedImages._2._3 / 4
    val encodedText = encodeText(sentences, List(imageTokenLength)).toArray

    (encodedText, preprocessedImages)
  }

  def tag(
      batch: Seq[Array[Int]],
      images: (org.intel.openvino.Tensor, (Int, Int, Int)),
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
      stopTokenIds: Array[Int],
      numOfCrops: Int = 16): Array[Array[Int]] = {

    val (pixelValues, (grid_t, grid_h, grid_w)) = images
    val imageGridTHW: Array[Array[Int]] = Array(Array(grid_t, grid_h, grid_w))
    val ignoreTokenIdsInt = ignoreTokenIds
    val expandedDecoderInputsVals = batch
    val sequencesLength = expandedDecoderInputsVals.map(x => x.length).toArray
    val maxSentenceLength = sequencesLength.max // - curLen
    //    val pixelValues = images._1
    //    val imageSizes = images._2
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

    val inferRequestImageEmbed =
      openvinoWrapper.get.imageEmbedding.getCompiledModel().create_infer_request()
    val inferRequestImageEmbedMerger =
      openvinoWrapper.get.imageEmbeddingMerger.getCompiledModel().create_infer_request()
    val inferRequestRotaryEmbedding =
      openvinoWrapper.get.rotaryEmbedding.getCompiledModel().create_infer_request()
    val inferRequestTextEmbedding =
      openvinoWrapper.get.textEmbedding.getCompiledModel().create_infer_request()
    val inferRequestMultimodalModelMerge =
      openvinoWrapper.get.multimodalMergeModel.getCompiledModel().create_infer_request()
    val inferRequestLanguageModel =
      openvinoWrapper.get.languageModel.getCompiledModel().create_infer_request()

    val generatedIds = generateGreedy(
      batch.toArray,
      batch.toArray,
      pixelValues,
      imageGridTHW,
      maxOutputLength,
      inferRequestImageEmbed,
      inferRequestImageEmbedMerger,
      inferRequestRotaryEmbedding,
      inferRequestTextEmbedding,
      inferRequestMultimodalModelMerge,
      inferRequestLanguageModel)
    generatedIds
  }

  def generateGreedy(
      encoderInputIds: Array[Array[Int]],
      decoderInputIds: Array[Array[Int]],
      pixelValues: org.intel.openvino.Tensor,
      imageGridTHW: Array[Array[Int]],
      maxOutputLength: Int,
      inferRequestImageEmbed: InferRequest,
      inferRequestImageEmbedMerger: InferRequest,
      inferRequestRotaryEmbedding: InferRequest,
      inferRequestTextEmbedding: InferRequest,
      inferRequestMultimodalModelMerge: InferRequest,
      inferRequestLanguageModel: InferRequest): Array[Array[Int]] = {

    var generatedIds: Array[Array[Int]] = Array()
    var decoderInputIdsCopied = decoderInputIds
    while (!greedyGenerationFinished(generatedIds, eosTokenId, maxOutputLength)) {
      val decoderOutputs = getModelOutputs(
        encoderInputIds,
        decoderInputIdsCopied,
        pixelValues,
        imageGridTHW,
        inferRequestImageEmbed,
        inferRequestImageEmbedMerger,
        inferRequestRotaryEmbedding,
        inferRequestTextEmbedding,
        inferRequestMultimodalModelMerge,
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

    val (encodedText, preprocessedImages) = encode(imageAnnotations, sentences, preprocessor)
//    val (pixelValues, imageSizes, imgTokens) = preprocessedImages
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
      pixelValues: org.intel.openvino.Tensor,
      imageGridTHW: Array[Array[Int]],
      inferRequestImageEmbed: InferRequest,
      inferRequestImageEmbedMerger: InferRequest,
      inferRequestRotaryEmbedding: InferRequest,
      inferRequestTextEmbedding: InferRequest,
      inferRequestMultimodalModelMerge: InferRequest,
      inferRequestLanguageModel: InferRequest): Array[Array[Float]] = {

    val imageEmbeddings = getImageEmbeddings(
      encoderInputIds,
      decoderInputIds,
      pixelValues,
      imageGridTHW,
      inferRequestImageEmbed,
      inferRequestImageEmbedMerger,
      inferRequestRotaryEmbedding,
      inferRequestTextEmbedding,
      inferRequestMultimodalModelMerge)

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
    val shape: Array[Int] = Array(3, 1, inputIdsLong.length / batchSize)

    val reshapedArray = Array(Array(inputPositionIDsLong))

    // Expand the array by replicating the first dimension
    val inputPositionIDsLongX3 =
      reshapedArray.map(x => Array(x, x, x)).flatten.flatten.flatten

    val decoderAttentionMask: org.intel.openvino.Tensor =
      new org.intel.openvino.Tensor(Array(batchSize, decoderInputIds.head.length), attentionMask)
    val decoderPositionIDs: org.intel.openvino.Tensor =
      new org.intel.openvino.Tensor(shape, inputPositionIDsLongX3)
    val beamIdxTensor: org.intel.openvino.Tensor =
      new org.intel.openvino.Tensor(Array(batchSize), beamIdx)

    val imgEmbeddingTensor =
      new org.intel.openvino.Tensor(imageEmbeddings.get_shape(), imageEmbeddings.data())

    inferRequestLanguageModel.set_tensor("inputs_embeds", imgEmbeddingTensor)
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

  def preprocessImage(
      imageAnnotations: Seq[AnnotationImage],
      preprocessor: Preprocessor,
      sizeFactor: Int = IMAGE_FACTOR,
      minPixels: Int = MIN_PIXELS,
      maxPixels: Int = MAX_PIXELS): (org.intel.openvino.Tensor, (Int, Int, Int)) = {

    val rescaledImage = imageAnnotations
      .map(annotations => {

        val (width, height) = smartResize(
          annotations.height,
          annotations.width,
          factor = sizeFactor,
          minPixels = MIN_PIXELS,
          maxPixels = MAX_PIXELS)

        val bufferedImage = ImageIOUtils.byteToBufferedImage(
          bytes = annotations.result,
          w = annotations.width,
          h = annotations.height,
          nChannels = annotations.nChannels)

        val resizedImage =
          ImageResizeUtils.resizeBufferedImage(height = height, width = width, resample = 3)(
            bufferedImage)

        val resizedDimensions = smartResize(
          resizedImage.getHeight,
          resizedImage.getWidth,
          factor = sizeFactor,
          minPixels = minPixels,
          maxPixels = maxPixels)

        val (resizedWidth, resizedHeight) = resizedDimensions

        val resizedImageArray = ImageResizeUtils.resizeBufferedImage(
          width = resizedWidth,
          height = resizedHeight,
          resample = 3)(resizedImage)

        val normalizedImage =
          ImageResizeUtils.normalizeAndConvertBufferedImage(
            img = resizedImageArray,
            mean = preprocessor.image_mean,
            std = preprocessor.image_std,
            doNormalize = preprocessor.do_normalize,
            doRescale = preprocessor.do_rescale,
            rescaleFactor = preprocessor.rescale_factor)

        normalizedImage
      })
      .toArray

    val inferRequestPatchReshape =
      openvinoWrapper.get.patchReshapeModel.getCompiledModel().create_infer_request()

    val patchTensor = new org.intel.openvino.Tensor(
      Array(
        rescaledImage.length,
        rescaledImage.head.length,
        rescaledImage.head.head.length,
        rescaledImage.head.head.head.length),
      rescaledImage.flatten.flatten.flatten.map(_.toFloat))

    // 2.0f if rescaledImage.length == 1 else 1.0f
    val factor: Long = if (rescaledImage.length == 1) 2L else 1L
    val repetitionFactorTensor = new org.intel.openvino.Tensor(Array[Int](), Array(factor))
    inferRequestPatchReshape.set_tensor("patches", patchTensor)
    inferRequestPatchReshape.set_tensor("repetition_factor", repetitionFactorTensor)

    inferRequestPatchReshape.infer()

    val pixel_values = inferRequestPatchReshape.get_output_tensor()
    val grid_t = if (rescaledImage.length == 1) 1 else Math.ceil(rescaledImage.length / 2).toInt
    val grid_h = (rescaledImage.head.head.length / 14).toInt
    val grid_w = (rescaledImage.head.head.head.length / 14).toInt
    (pixel_values, (grid_t, grid_h, grid_w))
  }

  def getImageEmbeddings(
      encoderInputIds: Array[Array[Int]],
      decoderInputIds: Array[Array[Int]],
      pixelValues: org.intel.openvino.Tensor,
      imageGridTHW: Array[Array[Int]],
      inferRequestImageEmbed: InferRequest,
      inferRequestImageEmbedMerger: InferRequest,
      inferRequestRotaryEmbedding: InferRequest,
      inferRequestTextEmbedding: InferRequest,
      inferRequestMultimodalModelMerge: InferRequest): org.intel.openvino.Tensor = {
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
          new org.intel.openvino.Tensor(pixelValues.get_shape(), pixelValues.data())
//
//        val pixelValuesTensor = pixelValues
        inferRequestImageEmbed.set_input_tensor(pixelValuesTensor)

        inferRequestImageEmbed.infer()

        val hiddenStates = inferRequestImageEmbed.get_output_tensor()

        val rotaryEmbeds = imageGridTHW.map(imageTHW => {
          val imageTHWTensor: org.intel.openvino.Tensor =
            new org.intel.openvino.Tensor(Array[Int](3), imageTHW.map(_.toLong))
          inferRequestRotaryEmbedding.set_input_tensor(imageTHWTensor)
          inferRequestRotaryEmbedding.infer()

          val rotary = inferRequestRotaryEmbedding.get_output_tensor()
          val rotaryData = rotary.data()
          (rotaryData, rotary.get_shape())
        })

        // rotary_pos_emb = torch.cat([torch.from_numpy(rotary_embedding(x)[0]) for x in image_grid_thw], dim=0)

        val rotaryPosEmb = rotaryEmbeds.flatMap(_._1)
        // shape should be batch_size x seq_len, hidden_size
        val rotaryShape =
          Array(rotaryEmbeds.length * rotaryEmbeds.head._2(0), rotaryEmbeds.head._2(1))
//        println("Rotary Shape: " + rotaryShape.mkString(","))
//        println("Rotary Pos Emb: " + rotaryPosEmb.length)
        val rotaryPosEmbTensor: org.intel.openvino.Tensor =
          new org.intel.openvino.Tensor(rotaryShape, rotaryPosEmb)

        // attention_mask = torch.zeros((1, hidden_states.shape[0], hidden_states.shape[0]), dtype=torch.bool)

        val attentionMask: Array[Float] =
          Array.fill(hiddenStates.get_shape()(0) * hiddenStates.get_shape()(0))(1f)

//        println("Hidden States Shape: " + hiddenStates.get_shape().mkString(","))
//        println("attentionMask Shape: " + attentionMask.length)

        val attentionMaskTensor: org.intel.openvino.Tensor =
          new org.intel.openvino.Tensor(
            Array(1, hiddenStates.get_shape()(0), hiddenStates.get_shape()(0)),
            attentionMask)

        inferRequestImageEmbedMerger.set_tensor("hidden_states", hiddenStates)
        inferRequestImageEmbedMerger.set_tensor("rotary_pos_emb", rotaryPosEmbTensor)
        inferRequestImageEmbedMerger.set_tensor("attention_mask", attentionMaskTensor)

        inferRequestImageEmbedMerger.infer()

        val imageEmbedMerged = inferRequestImageEmbedMerger.get_output_tensor()

        inferRequestTextEmbedding.set_input_tensor(inputIdsLongTensor)
        inferRequestTextEmbedding.infer()

        val textEmbeddings = inferRequestTextEmbedding.get_output_tensor()

        inferRequestMultimodalModelMerge.set_tensor("inputs_embeds", textEmbeddings)
        inferRequestMultimodalModelMerge.set_tensor("vision_embeds", imageEmbedMerged)
        inferRequestMultimodalModelMerge.set_tensor("input_ids", inputIdsLongTensor)

        inferRequestMultimodalModelMerge.infer()

        inferRequestMultimodalModelMerge.get_output_tensor()

      } else {
        inferRequestTextEmbedding.set_input_tensor(inputIdsLongTensor)
        inferRequestTextEmbedding.infer()

        inferRequestTextEmbedding.get_output_tensor()
      }
    imageEmbeddings
  }

}
