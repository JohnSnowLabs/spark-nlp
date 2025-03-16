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
import java.lang.Math

import com.johnsnowlabs.ml.ai.util.Generation.GenerationConfig
import com.johnsnowlabs.ml.onnx.OnnxWrapper.DecoderWrappers
import com.johnsnowlabs.ml.openvino.OpenvinoWrapper.JanusWrappers
import com.johnsnowlabs.nlp.annotators.common.Sentence
import com.johnsnowlabs.ml.util.{ONNX, Openvino}
import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common.SentenceSplit
import com.johnsnowlabs.nlp.annotators.cv.util.transform.ImageResizeUtils
import com.johnsnowlabs.nlp.annotators.cv.feature_extractor.Preprocessor
import com.johnsnowlabs.nlp.annotators.cv.util.io.ImageIOUtils
import com.johnsnowlabs.nlp.annotators.tokenizer.bpe.{BpeTokenizer, JanusTokenizer, SpecialTokens}
import org.intel.openvino.{InferRequest, Tensor}

import javax.imageio.ImageIO
import scala.util.Random
import scala.reflect.ClassTag
import java.awt.{Color, Graphics2D}
import java.awt.image.BufferedImage
import java.io.ByteArrayOutputStream
import scala.collection.JavaConverters._

private[johnsnowlabs] class Janus(
    val onnxWrappers: Option[DecoderWrappers],
    val openvinoWrapper: Option[JanusWrappers],
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

  val bpeTokenizer: JanusTokenizer = BpeTokenizer
    .forModel(
      "Janus",
      merges = merges,
      vocab = vocabulary,
      specialTokens = Some(specialTokens),
      addPrefixSpaceToSentence = true,
      alwaysAddPrefix = false)
    .asInstanceOf[JanusTokenizer]

  var randomSeedGenerator = new Random()

  /** Decode a sequence of sentences
    * @param sentences
    *   Sequence of sentences
    * @return
    *   Sequence of decoded sentences
    */
  def decode(sentences: Array[Array[Int]]): Seq[String] = {
    sentences.map(s => bpeTokenizer.decodeTokens(s.map(_.toInt)))
  }

  /** Encode a sequence of sentences for generation
    * @param sentences
    *   Sequence of sentences
    * @return
    *   Sequence of encoded sentences
    */
  private def encodeTextForGeneration(sentences: Seq[Annotation]): Seq[Array[Int]] = {
    val startOfImage = "<begin_of_image>"
    val endOfImage = "<end_of_image>"
    val startOfImageToken = vocabulary.getOrElse(startOfImage, 100016)
    val endOfImageToken = vocabulary.getOrElse(endOfImage, 100593)

    // encode text and add beginning of image token

    val tokens = SentenceSplit
      .unpack(sentences)
      .map(s => {
        val sentWithTask = s
        bpeTokenizer
          .tokenize(sentWithTask)
          .map(bpeTokenizer.encode)
          .flatMap(_.map(_.pieceId))
      })
      .map(s => Array(bosTokenId) ++ s ++ Array(startOfImageToken))

    tokens

  }

  /** Encode a sequence of sentences
    * @param sentences
    *   Sequence of sentences
    * @return
    *   Sequence of encoded sentences
    */
  def encodeText(sentences: Seq[Annotation], imgTokenLen: List[Int]): Seq[Array[Int]] = {

    val pattern = raw"<image_placeholder>".r

    val startOfImage = "<begin_of_image>"
    val endOfImage = "<end_of_image>"
    val startOfImageToken = vocabulary.getOrElse(startOfImage, 100016)
    val endOfImageToken = vocabulary.getOrElse(endOfImage, 100593)

    // raise an error if the pattern is not found in the text
    if (pattern.findFirstIn(sentences.head.result).isEmpty) {
      throw new IllegalArgumentException(
        "The pattern <image_placeholder> is not found in the text")
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
        val imgPaddingTokens =
          Array(startOfImageToken) ++ Array.fill(imgTokenLen)(imageToken) ++ Array(
            endOfImageToken)
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
      preprocessor: Preprocessor,
      imageTokenLength: Int = imageTokenLength)
      : (Seq[Array[Int]], Array[Array[Array[Array[Array[Float]]]]]) = {
    val preprocessedImages = encodeImage(imageAnnotations.toArray, preprocessor)
    val encodedText = encodeText(sentences, List(imageTokenLength)).toArray

    (encodedText, preprocessedImages)
  }

  def tag(
      batch: Seq[Array[Int]],
      images: Array[Array[Array[Array[Array[Float]]]]],
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

    val inferRequestLanguageModel =
      openvinoWrapper.get.languageModel.getCompiledModel().create_infer_request()
    val inferRequestVisionEmbeddingsModel =
      openvinoWrapper.get.visionEmbeddingsModel.getCompiledModel().create_infer_request()
    val inferRequestTextEmbeddingsModel =
      openvinoWrapper.get.textEmbeddingsModel.getCompiledModel().create_infer_request()
    val inferRequestLMHeadModel =
      openvinoWrapper.get.lmHeadModel.getCompiledModel().create_infer_request()
    val inferRequestMergeModel =
      openvinoWrapper.get.mergeModel.getCompiledModel().create_infer_request()

    val generatedIds = generateGreedy(
      batch.toArray,
      batch.toArray,
      pixelValues,
      maxOutputLength,
      inferRequestLanguageModel,
      inferRequestVisionEmbeddingsModel,
      inferRequestTextEmbeddingsModel,
      inferRequestLMHeadModel,
      inferRequestMergeModel)
    generatedIds
  }

  def generateGreedy(
      encoderInputIds: Array[Array[Int]],
      decoderInputIds: Array[Array[Int]],
      pixelValues: Array[Array[Array[Array[Array[Float]]]]],
      maxOutputLength: Int,
      inferRequestLanguageModel: InferRequest,
      inferRequestVisionEmbeddingsModel: InferRequest,
      inferRequestTextEmbeddingsModel: InferRequest,
      inferRequestLMHeadModel: InferRequest,
      inferRequestMergeModel: InferRequest): Array[Array[Int]] = {

    var generatedIds: Array[Array[Int]] = Array()
    var decoderInputIdsCopied = decoderInputIds
    while (!greedyGenerationFinished(generatedIds, eosTokenId, maxOutputLength)) {
      val decoderOutputs = getModelOutputs(
        encoderInputIds,
        decoderInputIdsCopied,
        pixelValues,
        inferRequestLanguageModel,
        inferRequestVisionEmbeddingsModel,
        inferRequestTextEmbeddingsModel,
        inferRequestLMHeadModel,
        inferRequestMergeModel)

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
      imageGenerateMode: Boolean,
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
      maxInputLength: Int,
      numOfParallelImages: Int): Seq[Annotation] = {

    if (imageGenerateMode) {
      randomSeedGenerator = randomSeed.map(s => new Random(s)).getOrElse(new Random())
      val encodedText: Array[Array[Int]] = encodeTextForGeneration(sentences).toArray
      val parallelSize = numOfParallelImages
      val tokens = Array.ofDim[Int](parallelSize * 2, encodedText.head.length)
      for (i <- 0 until parallelSize * 2) {
        if (i % 2 != 0) {
          tokens(i) = Array.fill(encodedText.head.length)(paddingTokenId)
          // update the first and last token to bos and eos respectively
          tokens(i)(0) = encodedText.head.head
          tokens(i)(encodedText.head.length - 1) = encodedText.head.last
        } else {
          tokens(i) = encodedText.head
        }
      }
      val generatedImages = generateImage(
        tokens,
        tokens,
        parallelSize = parallelSize,
        patchSize = 16,
        imageSize = preprocessor.size,
        randomSeed = randomSeed,
        inferRequestTextEmbeddingsModel =
          openvinoWrapper.get.textEmbeddingsModel.getCompiledModel().create_infer_request(),
        inferRequestGenEmbeddingsModel =
          openvinoWrapper.get.genEmbeddingsModel.getCompiledModel().create_infer_request(),
        inferRequestGenHeadModel =
          openvinoWrapper.get.genHeadModel.getCompiledModel().create_infer_request(),
        inferRequestLanguageModel =
          openvinoWrapper.get.languageModel.getCompiledModel().create_infer_request(),
        inferRequestGenDecoderModel =
          openvinoWrapper.get.genDecoderModel.getCompiledModel().create_infer_request())

      // group generated images into ( batch_size, parallel_size) and convert them to annotations
      val parallelSizeBatchedImages: Array[Array[BufferedImage]] =
        generatedImages.grouped(parallelSize).toArray

      val annotations = parallelSizeBatchedImages.zip(sentences).map { case (imgs, sent) =>
        var metadata = Map[String, String]()
        // add each image to the metadata
        imgs.zipWithIndex.foreach { case (img, i) =>
          val bos = new ByteArrayOutputStream()
          ImageIO.write(img, "png", bos)
          val base64EncodedImage = java.util.Base64.getEncoder.encodeToString(bos.toByteArray)
          metadata += (s"generated_image_$i" -> base64EncodedImage)
        }
        val annots = new Annotation(
          annotatorType = DOCUMENT,
          begin = 0,
          end = 0,
          result = sent.result,
          metadata = metadata)
        annots
      }
      annotations
    } else {
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
  }

  def getModelOutputs(
      encoderInputIds: Array[Array[Int]],
      decoderInputIds: Array[Array[Int]],
      pixelValues: Array[Array[Array[Array[Array[Float]]]]],
      inferRequestLanguageModel: InferRequest,
      inferRequestVisionEmbeddingsModel: InferRequest,
      inferRequestTextEmbeddingsModel: InferRequest,
      inferRequestLMHeadModel: InferRequest,
      inferRequestMergeModel: InferRequest): Array[Array[Float]] = {

    val mergeRequest = openvinoWrapper.get.mergeModel.getCompiledModel().create_infer_request()
    val inputEmbeds = getMultimodalEmbeddings(
      encoderInputIds,
      decoderInputIds,
      pixelValues,
      inferRequestVisionEmbeddingsModel,
      inferRequestTextEmbeddingsModel,
      mergeRequest)
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

    val result = inferRequestLanguageModel.get_tensor("last_hidden_state")

    inferRequestLMHeadModel.set_input_tensor(result)
    inferRequestLMHeadModel.infer()

    val logits = inferRequestLMHeadModel.get_output_tensor()

    val logitsRaw = logits.data()

    val sequenceLength = inputIdsLong.length / batchSize
    val decoderOutputs = (0 until batchSize).map(i => {
      logitsRaw
        .slice(
          i * sequenceLength * vocabSize + (sequenceLength - 1) * vocabSize,
          i * sequenceLength * vocabSize + sequenceLength * vocabSize)
    })
    decoderOutputs.toArray
  }

  def generateImage(
      encoderInputIds: Array[Array[Int]],
      decoderInputIds: Array[Array[Int]],
      parallelSize: Int = 1,
      patchSize: Int = 16,
      imageSize: Int = preprocessor.size,
      randomSeed: Option[Long] = None,
      inferRequestTextEmbeddingsModel: InferRequest,
      inferRequestGenEmbeddingsModel: InferRequest,
      inferRequestGenHeadModel: InferRequest,
      inferRequestLanguageModel: InferRequest,
      inferRequestGenDecoderModel: InferRequest): Array[BufferedImage] = {

    val generatedTokens = getImageModelOutputs(
      encoderInputIds,
      decoderInputIds,
      randomSeed,
      inferRequestTextEmbeddingsModel,
      inferRequestGenEmbeddingsModel,
      inferRequestGenHeadModel,
      inferRequestLanguageModel)

    inferRequestGenDecoderModel.set_tensor(
      "code_b",
      new org.intel.openvino.Tensor(
        Array(generatedTokens.length, generatedTokens.head.length),
        generatedTokens.flatten.map(_.toLong)))

    inferRequestGenDecoderModel.set_tensor(
      "shape",
      new org.intel.openvino.Tensor(
        Array(4),
        Array(parallelSize, 8, imageSize / patchSize, imageSize / patchSize).map(_.toLong)))

    inferRequestGenDecoderModel.infer()

    val dec = inferRequestGenDecoderModel.get_output_tensor()

    val decShape = dec.get_shape()
    val decChannelsLast = transposeArray(dec.data(), decShape, Array(0, 2, 3, 1))

    val decChannelsLastReshaped =
      reshape4D(decChannelsLast, decShape(0), decShape(2), decShape(3), decShape(1))

    val decClipped: Array[Array[Array[Array[Int]]]] = decChannelsLastReshaped.map { x =>
      x.map { y =>
        y.map { z =>
          z.map { w =>
            Math.min(Math.max(((w + 1) / 2) * 255, 0), 255).toInt
          }
        }
      }
    }

    // convert each image to a BufferedImage
    val bufferedImages = decClipped.map { img =>
      ImageIOUtils.arrayToBufferedImage(img)
    }
    bufferedImages
  }

  def getImageModelOutputs(
      encoderInputIds: Array[Array[Int]],
      decoderInputIds: Array[Array[Int]],
      randomSeed: Option[Long] = None,
      inferRequestTextEmbeddingsModel: InferRequest,
      inferRequestGenEmbeddingsModel: InferRequest,
      inferRequestGenHeadModel: InferRequest,
      inferRequestLanguageModel: InferRequest): Array[Array[Int]] = {

    var generatedTokens: Array[Array[Int]] = Array()
    var nextInputEmbedsTensor: Option[org.intel.openvino.Tensor] = None
    var decoderInputIdsCopied = decoderInputIds.clone()
    // run the model for imageTokenLength times
    for (i <- 0 until imageTokenLength) {
      val nextTokenIds = getNextImageTokens(
        encoderInputIds,
        decoderInputIdsCopied,
        cfgWeight = 5.0f,
        temperature = 1.0f,
        randomSeed = randomSeed,
        inputEmbeds = nextInputEmbedsTensor,
        inferRequestTextEmbeddingsModel,
        inferRequestGenHeadModel,
        inferRequestLanguageModel)
      val nextTokenIdsTensor = new org.intel.openvino.Tensor(
        Array(nextTokenIds.length * 2),
        nextTokenIds.flatMap(x => Array(x, x)).map(_.toLong))

      inferRequestGenEmbeddingsModel.set_input_tensor(nextTokenIdsTensor)
      inferRequestGenEmbeddingsModel.infer()

      val imageEmbeddings = inferRequestGenEmbeddingsModel.get_output_tensor()

      nextInputEmbedsTensor = None
      nextInputEmbedsTensor = Some(
        new org.intel.openvino.Tensor(
          Array(imageEmbeddings.get_shape()(0), 1, imageEmbeddings.get_shape()(1)),
          imageEmbeddings.data()))

      if (generatedTokens.isEmpty) {
        generatedTokens = nextTokenIds.map(Array(_))
      } else {
        generatedTokens =
          generatedTokens.zip(nextTokenIds).map { case (currentIds: Array[Int], nextId: Int) =>
            currentIds ++ Array(nextId)
          }
      }

      // repeat the nextTokenIds twice and add them to the decoder input ids
      val repeatedNextTokenIds = nextTokenIds.flatMap(x => Array(x, x))

      // extend decoder input ids to include the generated tokens. Decoder input ids are duplicated for each image
      decoderInputIdsCopied =
        decoderInputIdsCopied.zip(repeatedNextTokenIds).map { case (currentIds, nextId) =>
          currentIds ++ Array(nextId)
        }
    }
    generatedTokens
  }

  private def getNextImageTokens(
      encoderInputIds: Array[Array[Int]],
      decoderInputIds: Array[Array[Int]],
      cfgWeight: Float = 5.0f,
      temperature: Float = 1.0f,
      randomSeed: Option[Long] = None,
      inputEmbeds: Option[Tensor],
      inferRequestTextEmbeddingsModel: InferRequest,
      inferRequestGenHeadModel: InferRequest,
      inferRequestLanguageModel: InferRequest): Array[Int] = {

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

    val inputEmbedsTensor: org.intel.openvino.Tensor = if (inputEmbeds.isDefined) {
      inputEmbeds.get
    } else {
      val inputIdsLongTensor: org.intel.openvino.Tensor =
        new org.intel.openvino.Tensor(shape, inputIdsLong)
      inferRequestTextEmbeddingsModel.set_input_tensor(inputIdsLongTensor)
      inferRequestTextEmbeddingsModel.infer()

      val textEmbeddings = inferRequestTextEmbeddingsModel.get_output_tensor()
      textEmbeddings
    }

    inferRequestLanguageModel.set_tensor("inputs_embeds", inputEmbedsTensor)
    inferRequestLanguageModel.set_tensor("attention_mask", decoderAttentionMask)
    inferRequestLanguageModel.set_tensor("position_ids", decoderPositionIDs)
    inferRequestLanguageModel.set_tensor("beam_idx", beamIdxTensor)

    inferRequestLanguageModel.infer()

    val result = inferRequestLanguageModel.get_tensor("last_hidden_state")
    val resultShape = result.get_shape()
    // select the last hidden state
    // (2*parallel_images, sequence_length, hidden_size)
    // Reshape the tensor
    val reshapedArray: Array[Array[Array[Float]]] =
      reshape3D(result.data(), resultShape(0), resultShape(1), resultShape(2))
    val lastResult = reshapedArray.map { x =>
      x(resultShape(1) - 1)
    }.toArray
    val lastResultTensor =
      new org.intel.openvino.Tensor(Array(resultShape(0), resultShape(2)), lastResult.flatten)

    inferRequestGenHeadModel.set_input_tensor(lastResultTensor)
    inferRequestGenHeadModel.infer()

    val logits = inferRequestGenHeadModel.get_output_tensor()
    val logitsShape = logits.get_shape()

    val logitsRaw = logits.data()
    val reshapedLogits: Array[Array[Float]] =
      reshape2D(logitsRaw, logitsShape(0), logitsShape(1))
    // every second element starting from 0 to the end will be the conditional logits\
    val logitCond = reshapedLogits.zipWithIndex.filter(_._2 % 2 == 0).map(_._1)
    // every second element starting from 1 to the end will be the unconditional logits
    val logitUncond = reshapedLogits.zipWithIndex.filter(_._2 % 2 == 1).map(_._1)

    val logitDiff = logitCond.zip(logitUncond).map { case (cond, uncond) =>
      cond.zip(uncond).map { case (c, u) =>
        u + cfgWeight * (c - u)
      }
    }

    val probs = logitDiff.map(softmax)
    val nextTokenIds = multinomial(probs, numSamples = 1, seed = randomSeed)
    // pick a random token from the nextTokenIds
//    val randomIndex = new Random()
//    nextTokenIds.map(x => x(randomIndex.nextInt(x.length)))
    nextTokenIds.map(_.head)

  }

  private def multinomial(
      probs: Array[Array[Float]],
      numSamples: Int = 1,
      seed: Option[Long] = None): Array[Array[Int]] = {
    val random = seed.map(s => new Random(s)).getOrElse(new Random())
    probs.map { p =>
      require(p.nonEmpty, "Probability array cannot be empty")
      require(p.forall(_ >= 0.0f), "Probabilities must be non-negative")
      require(Math.abs(p.sum - 1.0f) < 1e-3, "Probabilities must sum to approximately 1.0")
      require(p.exists(_ > 0.0f), "Probability array cannot contain all zeros")

      val cumSum = p.scanLeft(0.0f)(_ + _).drop(1)

      (0 until numSamples).map { _ =>
        val rand = Math.nextAfter(random.nextFloat(), Float.PositiveInfinity)
        cumSum.indexWhere(_ > rand) match {
          case -1 => cumSum.length - 1 // Ensure a valid index is always chosen
          case idx => idx
        }
      }.toArray
    }.toArray
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

  def getResizeSizes(
      width: Int,
      height: Int,
      minSize: Int = 14,
      imageSize: Int = 384): (Int, Int) = {
    val maxSize = math.max(width, height)
    (
      math.max((height.toFloat / maxSize * imageSize).toInt, minSize),
      math.max((width.toFloat / maxSize * imageSize).toInt, minSize))
  }

  def expandToSquare(img: BufferedImage, r: Int, g: Int, b: Int): BufferedImage = {
    val backgroundColor = new Color(r, g, b)
    val width = img.getWidth
    val height = img.getHeight

    if (width == height) {
      img
    } else {
      val size = Math.max(width, height)
      val squaredImage = new BufferedImage(size, size, img.getType)
      val g2d: Graphics2D = squaredImage.createGraphics()

      // Fill the background
      g2d.setColor(backgroundColor)
      g2d.fillRect(0, 0, size, size)

      // Calculate the position to center the original image
      val x = if (width < height) (size - width) / 2 else 0
      val y = if (height < width) (size - height) / 2 else 0

      // Draw the original image onto the new square image
      g2d.drawImage(img, x, y, null)
      g2d.dispose()

      squaredImage
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

      val (resize_height, resize_width): (Int, Int) = getResizeSizes(
        width = bufferedImage.getWidth,
        height = bufferedImage.getHeight,
        imageSize = preprocessor.size)

      val resizedImage = if (preprocessor.do_resize) {
        ImageResizeUtils.resizeBufferedImage(
          width = resize_height,
          height = resize_width,
          preprocessor.resample)(bufferedImage)
      } else bufferedImage

      val resizedImageSquare = expandToSquare(
        resizedImage,
        (preprocessor.image_mean(0) * 255).toInt,
        (preprocessor.image_mean(1) * 255).toInt,
        (preprocessor.image_mean(2) * 255).toInt)

      val normalizedImage =
        ImageResizeUtils.normalizeAndConvertBufferedImage(
          img = resizedImageSquare,
          mean = preprocessor.image_mean,
          std = preprocessor.image_std,
          doNormalize = preprocessor.do_normalize,
          doRescale = preprocessor.do_rescale,
          rescaleFactor = preprocessor.rescale_factor)

      Array(normalizedImage)
    }

    batchProcessedImages

  }

  def getMultimodalEmbeddings(
      encoderInputIds: Array[Array[Int]],
      decoderInputIds: Array[Array[Int]],
      pixelValues: Array[Array[Array[Array[Array[Float]]]]],
      inferRequestVisionEmbeddingsModel: InferRequest,
      inferRequestTextEmbeddingsModel: InferRequest,
      inferRequestMergeModel: InferRequest): org.intel.openvino.Tensor = {
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
            Array(
              pixelValues.length,
              pixelValues.head.length,
              pixelValues.head.head.length,
              pixelValues.head.head.head.length,
              pixelValues.head.head.head.head.length),
            pixelValues.flatten.flatten.flatten.flatten.map(_.toFloat))

        // Get image embeddings
        inferRequestVisionEmbeddingsModel.set_input_tensor(pixelValuesTensor)

        inferRequestVisionEmbeddingsModel.infer()

        val imageEmbeddings = inferRequestVisionEmbeddingsModel.get_output_tensor()

        // Get text embeddings
        inferRequestTextEmbeddingsModel.set_input_tensor(inputIdsLongTensor)

        inferRequestTextEmbeddingsModel.infer()

        val textEmbeddings = inferRequestTextEmbeddingsModel.get_output_tensor()

        // Merge image and text embeddings
        inferRequestMergeModel.set_tensor("vision_embeds", imageEmbeddings)
        inferRequestMergeModel.set_tensor("inputs_embeds", textEmbeddings)
        inferRequestMergeModel.set_tensor("input_ids", inputIdsLongTensor)

        inferRequestMergeModel.infer()

        inferRequestMergeModel.get_tensor("final_embeddings")
      } else {
        // Get text embeddings
        inferRequestTextEmbeddingsModel.set_input_tensor(inputIdsLongTensor)

        inferRequestTextEmbeddingsModel.infer()

        val textEmbeddings = inferRequestTextEmbeddingsModel.get_output_tensor()

        textEmbeddings
      }
    imageEmbeddings
  }

  def softmax(logitValues: Array[Float]): Array[Float] = {
    val maxLogit = logitValues.max
    val logitsExp = logitValues.map(l => Math.exp(l - maxLogit))
    val expSum = logitsExp.sum
    logitsExp.map(exp => (exp / expSum).toFloat)
  }

  // logSoftmax
  def logSoftmax(logitValues: Array[Float]): Array[Float] = {
    val maxLogit = logitValues.max
    val logitsExp = logitValues.map(l => Math.exp(l - maxLogit))
    val expSum = logitsExp.sum
    val logSumExp = Math.log(expSum)
    logitValues.map(l => l - maxLogit - logSumExp).map(_.toFloat)
  }

  // Function to reshape the flattened array
  def reshapeArray(flatArray: Array[Float], shape: Array[Int]): Any = {
    require(flatArray.length == shape.product, "Shape does not match data length")

    def recursiveReshape(data: Array[Float], shape: List[Int]): Any = shape match {
      case Nil => data.head // Base case: return a single element
      case head :: Nil => data.grouped(head).toArray.asInstanceOf[Array[Any]] // 1D array
      case head :: tail =>
        data
          .grouped(head)
          .map(subArr => recursiveReshape(subArr, tail))
          .toArray
          .asInstanceOf[Array[Any]] // Cast to Array[Any] to preserve structure
    }

    recursiveReshape(flatArray, shape.toList).asInstanceOf[Array[Any]]
  }

  def reshape2D(data: Array[Float], rows: Int, cols: Int): Array[Array[Float]] = {
//    data.grouped(cols).toArray.map(_.toArray)
//    i * sequenceLength * vocabSize + (sequenceLength - 1) * vocabSize,
//    i * sequenceLength * vocabSize + sequenceLength * vocabSize)
    0.until(rows)
      .map { i =>
        data.slice(i * cols, (i + 1) * cols)
      }
      .toArray
  }

  def reshape3D(
      data: Array[Float],
      depth: Int,
      rows: Int,
      cols: Int): Array[Array[Array[Float]]] = {
//    data.grouped(rows * cols).toArray.map { slice =>
//      reshape2D(slice, rows, cols)
//    }
    // use the depth to slice the data
    0.until(depth)
      .map { i =>
        data.slice(i * rows * cols, (i + 1) * rows * cols)
      }
      .map { slice =>
        reshape2D(slice, rows, cols)
      }
      .toArray
  }

  def reshape4D(
      data: Array[Float],
      batch: Int,
      depth: Int,
      rows: Int,
      cols: Int): Array[Array[Array[Array[Float]]]] = {
    data.grouped(depth * rows * cols).toArray.map { slice =>
      reshape3D(slice, depth, rows, cols)
    }
  }

  def transposeArray[T: ClassTag](
      inputArray: Array[T],
      inputArrayShape: Array[Int],
      axes: Array[Int]): Array[T] = {
    require(
      inputArrayShape.length == axes.length,
      "Axes must have the same length as the shape dimensions")

    val outputShape = axes.map(inputArrayShape(_))
    val size = inputArray.length
    val inputStrides = inputArrayShape.scanRight(1)(_ * _).tail
    val outputStrides = outputShape.scanRight(1)(_ * _).tail

    def getTransposedIndex(index: Int): Int = {
      val originalIndices =
        inputArrayShape.indices.map(i => (index / inputStrides(i)) % inputArrayShape(i))
      val transposedIndices = axes.map(originalIndices)
      transposedIndices.zip(outputStrides).map { case (idx, stride) => idx * stride }.sum
    }

    val outputArray = new Array[T](size)
    for (i <- inputArray.indices) {
      outputArray(getTransposedIndex(i)) = inputArray(i)
    }
    outputArray
  }

}
