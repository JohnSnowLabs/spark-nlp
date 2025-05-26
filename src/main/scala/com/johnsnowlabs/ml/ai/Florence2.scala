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

import com.johnsnowlabs.ml.ai.util.Generation.GenerationConfig
import com.johnsnowlabs.ml.onnx.OnnxWrapper.DecoderWrappers
import com.johnsnowlabs.ml.openvino.OpenvinoWrapper.Florence2Wrappers
import com.johnsnowlabs.ml.util.{ONNX, Openvino}
import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common.{Sentence, SentenceSplit}
import com.johnsnowlabs.nlp.annotators.cv.feature_extractor.Preprocessor
import com.johnsnowlabs.nlp.annotators.cv.util.io.ImageIOUtils
import com.johnsnowlabs.nlp.annotators.cv.util.transform.ImageResizeUtils
import com.johnsnowlabs.nlp.annotators.tokenizer.bpe.{
  BpeTokenizer,
  Florence2Tokenizer,
  SpecialTokens
}
import org.intel.openvino.InferRequest
import com.johnsnowlabs.ml.ai.util.Florence2Utils
import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.json4s.JsonDSL._

private[johnsnowlabs] class Florence2(
    val onnxWrappers: Option[DecoderWrappers],
    val openvinoWrapper: Option[Florence2Wrappers],
    merges: Map[(String, String), Int],
    vocabulary: Map[String, Int],
    addedTokens: Map[String, Int],
    preprocessor: Preprocessor,
    generationConfig: GenerationConfig)
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

  val bpeTokenizer: Florence2Tokenizer = BpeTokenizer
    .forModel(
      "florence2",
      merges = merges,
      vocab = vocabulary,
      specialTokens = Some(specialTokens),
      addPrefixSpaceToSentence = false,
      alwaysAddPrefix = false,
      prependString = "")
    .asInstanceOf[Florence2Tokenizer]

  val decoderModel = openvinoWrapper.get.decoderModel.getCompiledModel()
  val encoderModel = openvinoWrapper.get.encoderModel.getCompiledModel()
  val imageEmbedModel = openvinoWrapper.get.imageEmbedModel.getCompiledModel()
  val textEmbeddingsModel = openvinoWrapper.get.textEmbeddingsModel.getCompiledModel()
  val modelMergerModel = openvinoWrapper.get.modelMergerModel.getCompiledModel()

  var inferRequestDecoderModel = decoderModel.create_infer_request()
  var inferRequestEncoderModel = encoderModel.create_infer_request()
  var inferRequestImageEncoder = imageEmbedModel.create_infer_request()
  var inferRequestTextEmbeddings = textEmbeddingsModel.create_infer_request()
  var inferRequestModelMerger = modelMergerModel.create_infer_request()

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

    SentenceSplit
      .unpack(sentences)
      .map(s => {
        val sentWithTask = s
        bpeTokenizer
          .tokenize(sentWithTask)
          .map(bpeTokenizer.encode)
          .flatMap(_.map(_.pieceId))
      })
  }

  def encode(
      imageAnnotations: Seq[AnnotationImage],
      sentences: Seq[Annotation],
      preprocessor: Preprocessor): (Seq[Array[Int]], Array[Array[Array[Array[Float]]]]) = {
    val preprocessedImages = encodeImage(imageAnnotations.toArray, preprocessor)
    val encodedText = encodeText(sentences).toArray

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


//    use eosTokenId as the starting token for the decoder
    val decoderInputIds =
      expandedDecoderInputsVals.map { _ => Array(eosTokenId) }

    val generatedIds = generateGreedy(
      batch.toArray,
      decoderInputIds.toArray,
      pixelValues,
      maxOutputLength,
      inferRequestDecoderModel,
      inferRequestEncoderModel,
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
      inferRequestDecoderModel: InferRequest,
      inferRequestEncoderModel: InferRequest,
      inferRequestImageEncoder: InferRequest,
      inferRequestTextEmbeddings: InferRequest,
      inferRequestModelMerger: InferRequest): Array[Array[Int]] = {

    var generatedIds: Array[Array[Int]] = Array()
    var decoderInputIdsCopied = decoderInputIds

    val (encoderLastState, encoderAttentionMask) = getEncoderOutput(
      encoderInputIds,
      pixelValues,
      inferRequestImageEncoder,
      inferRequestEncoderModel,
      inferRequestTextEmbeddings,
      inferRequestModelMerger)

    val encoderAttentionMaskLong = new org.intel.openvino.Tensor(
      encoderAttentionMask.get_shape(),
      Array.fill(encoderAttentionMask.get_shape()(0) * encoderAttentionMask.get_shape()(1))(1L))

    val encoderLastStateLong =
      new org.intel.openvino.Tensor(encoderLastState.get_shape(), encoderLastState.data())
    (encoderLastStateLong, encoderAttentionMaskLong)

    while (!greedyGenerationFinished(generatedIds, eosTokenId, maxOutputLength)) {
      val decoderOutputs = getModelOutputs(
        decoderInputIdsCopied,
        encoderLastState,
        encoderAttentionMaskLong,
        inferRequestDecoderModel)

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
    this.synchronized {
      // Preprocessing: map tasks to prompts
      val inputTexts = sentences.map(_.result)
      val prompts = Florence2Utils.constructPrompts(inputTexts)
      val promptAnnotations = prompts.zip(sentences).map { case (prompt, ann) =>
        // add task to metadata
        ann.copy(result = prompt, metadata = ann.metadata ++ Map("task" -> ann.result))
      }
      val (encodedText, preprocessedImages) =
        encode(imageAnnotations, promptAnnotations, preprocessor)

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
      val annotations = decoded.zip(promptAnnotations).map { case (content, ann) =>
        nextSentEnd += content.length - 1
        // Get the task from the annotation metadata, default to <CAPTION>
        val task = ann.metadata.getOrElse("task", "<CAPTION>")
        // Use image size from the first image annotation if available, else (1000, 1000)
        val imageSize =
          imageAnnotations.headOption.map(img => (img.width, img.height)).getOrElse((1000, 1000))
        val postProcessed = Florence2Utils.postProcessGeneration(content, task, imageSize)
        // Serialize postProcessed to JSON string for raw values using json4s
        implicit val formats = DefaultFormats
        val postProcessedRaw = postProcessed match {
          case Florence2Utils.BBoxesResult(bboxes) =>
            compact(render(Extraction.decompose(bboxes)))
          case Florence2Utils.OCRResult(instances) =>
            compact(render(Extraction.decompose(instances)))
          case Florence2Utils.PhraseGroundingResult(instances) =>
            compact(render(Extraction.decompose(instances)))
          case Florence2Utils.PolygonsResult(instances) =>
            compact(render(Extraction.decompose(instances)))
          case Florence2Utils.MixedResult(bboxes, bboxesLabels, polygons, polygonsLabels) =>
            val obj = ("bboxes" -> bboxes.map(_.bbox)) ~
              ("bboxesLabels" -> bboxesLabels) ~
              ("polygons" -> polygons) ~
              ("polygonsLabels" -> polygonsLabels)
            compact(render(obj))
          case Florence2Utils.PureTextResult(text) =>
            compact(render("text" -> text))
        }
        // If we have an image, try to generate a visualization
        val imageOpt = imageAnnotations.headOption.map { imgAnn =>
          com.johnsnowlabs.nlp.annotators.cv.util.io.ImageIOUtils
            .byteToBufferedImage(imgAnn.result, imgAnn.width, imgAnn.height, imgAnn.nChannels)
        }
        val imageBase64Opt = imageOpt.flatMap { img =>
          Florence2Utils.postProcessImage(img, task, postProcessed)
        }
        val newMetadata =
          ann.metadata ++
            Map(
              "florence2_postprocessed" -> postProcessed.toString,
              "florence2_postprocessed_raw" -> postProcessedRaw) ++
            imageBase64Opt.map(b64 => Map("florence2_image" -> b64)).getOrElse(Map.empty)
        new Annotation(
          annotatorType = DOCUMENT,
          begin = sentBegin,
          end = nextSentEnd,
          result = content,
          metadata = newMetadata)
      }
      // reset requests
      inferRequestDecoderModel.release()
      openvinoWrapper.get.decoderModel.deleteCompiledModel()
      inferRequestDecoderModel =
        openvinoWrapper.get.decoderModel.getCompiledModel().create_infer_request()

      annotations
    }
  }

  def getModelOutputs(
      decoderInputIds: Array[Array[Int]],
      encoderLastState: org.intel.openvino.Tensor,
      encoderAttentionMask: org.intel.openvino.Tensor,
      inferRequestDecoderModel: InferRequest): Array[Array[Float]] = {

    val (inputIdsLong, inputPositionIDsLong): (Array[Long], Array[Long]) = {
      // Subsequent passes
      val inpIdsLong = decoderInputIds.map { tokenIds => tokenIds.last.toLong }
      val posIdsLong = decoderInputIds.map { tokenIds =>
        tokenIds.zipWithIndex.map { case (_, i) =>
          i.toLong
        }.last
      }
      (inpIdsLong, posIdsLong)
    }

    val batchSize: Int = decoderInputIds.length
    val beamIdx: Array[Int] = {
      val beamIdx = new Array[Int](batchSize)
      for (i <- 0 until batchSize) {
        beamIdx(i) = i
      }
      beamIdx
    }
    val shape: Array[Int] = Array(batchSize, inputIdsLong.length / batchSize)
    val beamIdxTensor: org.intel.openvino.Tensor =
      new org.intel.openvino.Tensor(Array(batchSize), beamIdx)
    val inputIdsLongTensor: org.intel.openvino.Tensor =
      new org.intel.openvino.Tensor(shape, inputIdsLong)

    inferRequestDecoderModel.set_tensor("encoder_hidden_states", encoderLastState)
    inferRequestDecoderModel.set_tensor("encoder_attention_mask", encoderAttentionMask)
    inferRequestDecoderModel.set_tensor("decoder_input_ids", inputIdsLongTensor)
    inferRequestDecoderModel.set_tensor("beam_idx", beamIdxTensor)

    inferRequestDecoderModel.start_async()
    inferRequestDecoderModel.wait_async()
    val result = inferRequestDecoderModel.get_output_tensor()

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

  def getEncoderOutput(
      encoderInputIds: Array[Array[Int]],
      pixelValues: Array[Array[Array[Array[Float]]]],
      inferRequestImageEncoder: InferRequest,
      inferRequestEncoder: InferRequest,
      inferRequestTextEmbeddings: InferRequest,
      inferRequestModelMerger: InferRequest)
      : (org.intel.openvino.Tensor, org.intel.openvino.Tensor) = {
    val inputIdsLong: Array[Long] = {
      // First pass
      val inpIdsLong = encoderInputIds.flatMap { tokenIds => tokenIds.map(_.toLong) }
      inpIdsLong
    }
    val batchSize: Int = encoderInputIds.length
    val shape: Array[Int] = Array(batchSize, inputIdsLong.length / batchSize)
    val inputIdsLongTensor: org.intel.openvino.Tensor =
      new org.intel.openvino.Tensor(shape, inputIdsLong)

    val inputEmbeddings: org.intel.openvino.Tensor = {
      val pixelValuesTensor: org.intel.openvino.Tensor =
        new org.intel.openvino.Tensor(
          Array(batchSize, 3, preprocessor.size, preprocessor.size),
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

      inferRequestModelMerger.infer()
      inferRequestModelMerger.get_tensor("final_embedding")
    }
    val attentionMaskShape: Array[Int] =
      Array(inputEmbeddings.get_shape()(0), inputEmbeddings.get_shape()(1))
    val encoderAttentionMask: Array[Float] =
      Array.fill(inputEmbeddings.get_shape()(0) * inputEmbeddings.get_shape()(1))(1.0f)
    val encoderAttentionMaskTensor: org.intel.openvino.Tensor =
      new org.intel.openvino.Tensor(attentionMaskShape, encoderAttentionMask)
    val encoderOutput: org.intel.openvino.Tensor = {
      inferRequestEncoder.set_tensor("inputs_embeds", inputEmbeddings)
      inferRequestEncoder.set_tensor("attention_mask", encoderAttentionMaskTensor)
      inferRequestEncoder.infer()
      val encoderOutput = inferRequestEncoder.get_tensor("last_hidden_state")
      encoderOutput
    }
    (encoderOutput, encoderAttentionMaskTensor)
  }
}
