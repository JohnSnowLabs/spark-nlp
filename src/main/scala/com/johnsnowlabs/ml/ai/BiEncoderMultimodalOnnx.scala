/*
 * Copyright 2017-2026 John Snow Labs
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

import ai.onnxruntime.OnnxTensor
import com.johnsnowlabs.ml.onnx.{OnnxSession, OnnxWrapper}
import com.johnsnowlabs.nlp.annotators.common.Sentence
import com.johnsnowlabs.nlp.annotators.cv.feature_extractor.Preprocessor
import com.johnsnowlabs.nlp.annotators.cv.util.io.ImageIOUtils
import com.johnsnowlabs.nlp.annotators.cv.util.transform.ImageResizeUtils
import com.johnsnowlabs.nlp.annotators.cv.util.transform.Qwen2VLUtils.smartResize
import com.johnsnowlabs.nlp.annotators.tokenizer.bpe.{
  BpeTokenizer,
  Qwen2VLTokenizer,
  SpecialTokens
}
import com.johnsnowlabs.nlp.{Annotation, AnnotationImage}

import scala.collection.JavaConverters._

private[johnsnowlabs] final class BiEncoderMultimodalOnnx(
    val textOnnxWrapper: OnnxWrapper,
    val imageOnnxWrapper: OnnxWrapper,
    vocabulary: Map[String, Int],
    merges: Map[(String, String), Int],
    addedTokens: Map[String, Int],
    preprocessor: Preprocessor,
    bosTokenId: Int,
    eosTokenId: Int,
    padTokenId: Int,
    imageTokenId: Int,
    spatialMergeSize: Int,
    patchSize: Int,
    temporalPatchSize: Int,
    minPixels: Int,
    maxPixels: Int,
    instruction: String,
    imagePromptFixTokenIds: Array[Int])
    extends BiEncoderMultimodal
    with Serializable {

  private val onnxSessionOptions: Map[String, String] = new OnnxSession().getSessionOptions

  private val reversedVocabulary: Map[Int, String] = vocabulary.map(_.swap)

  private val specialTokens = SpecialTokens(
    vocabulary,
    startTokenString = reversedVocabulary(bosTokenId),
    endTokenString = reversedVocabulary(eosTokenId),
    unkTokenString = reversedVocabulary(padTokenId),
    maskTokenString = reversedVocabulary(padTokenId),
    padTokenString = reversedVocabulary(padTokenId),
    additionalStrings = addedTokens.keys.toArray)

  private val tokenizer = BpeTokenizer
    .forModel(
      "qwen2vl",
      merges = merges,
      vocab = vocabulary,
      specialTokens = Some(specialTokens),
      addPrefixSpaceToSentence = false,
      alwaysAddPrefix = false,
      prependString = "")
    .asInstanceOf[Qwen2VLTokenizer]

  override def predict(
      documentAnnotations: Seq[Annotation],
      imageAnnotations: Seq[AnnotationImage]): Seq[BiEncoderEmbeddingPair] = {
    require(
      documentAnnotations.length == imageAnnotations.length,
      s"BiEncoderMultimodalOnnx requires aligned text and image inputs. Found ${documentAnnotations.length} text annotations and ${imageAnnotations.length} image annotations.")

    if (documentAnnotations.isEmpty) {
      Seq.empty
    } else {
      val textEmbeddings = predictText(documentAnnotations)
      val imageEmbeddings = predictImage(imageAnnotations)
      textEmbeddings.zip(imageEmbeddings).map { case (textEmbedding, imageEmbedding) =>
        BiEncoderEmbeddingPair(textEmbedding, imageEmbedding)
      }
    }
  }

  private def predictText(documentAnnotations: Seq[Annotation]): Seq[Array[Float]] = {
    val prompts =
      documentAnnotations.map(annotation => encodeTextPrompt(buildTextPrompt(annotation)))
    val (inputIds, attentionMask) = leftPad(prompts)
    runModel(textOnnxWrapper, Map("input_ids" -> inputIds, "attention_mask" -> attentionMask))
  }

  private def predictImage(imageAnnotations: Seq[AnnotationImage]): Seq[Array[Float]] = {
    val imageInputs = imageAnnotations.map(buildImageTowerInput)
    val (inputIds, attentionMask) = leftPad(imageInputs.map(_.inputIds))
    val pixelValues = imageInputs.flatMap(_.pixelValues).toArray
    val imageGridThw = imageInputs.map(_.imageGridThw).toArray

    runModel(
      imageOnnxWrapper,
      Map(
        "input_ids" -> inputIds,
        "attention_mask" -> attentionMask,
        "pixel_values" -> pixelValues,
        "image_grid_thw" -> imageGridThw))
  }

  private def runModel(wrapper: OnnxWrapper, rawInputs: Map[String, Any]): Seq[Array[Float]] = {
    val (session, env) = wrapper.getSession(onnxSessionOptions)
    val tensors = rawInputs.map { case (name, value) =>
      name -> OnnxTensor.createTensor(env, value)
    }

    val results = session.run(tensors.asJava)
    try {
      val outputName = session.getOutputInfo.asScala.keys.toSeq.sorted.headOption.getOrElse {
        throw new IllegalStateException("No ONNX outputs found for multimodal embedding model.")
      }

      val outputTensor = results.get(outputName).get().asInstanceOf[OnnxTensor]
      val outputShape = outputTensor.getInfo.getShape
      require(
        outputShape.nonEmpty,
        s"Unexpected output shape ${outputShape.mkString("[", ", ", "]")} for multimodal embedding model.")

      val embeddingDim = outputShape.last.toInt
      val buffer = outputTensor.getFloatBuffer
      val values = new Array[Float](buffer.remaining())
      buffer.get(values)
      values.grouped(embeddingDim).toSeq
    } finally {
      results.close()
      tensors.values.foreach(_.close())
    }
  }

  private def buildTextPrompt(annotation: Annotation): String = {
    s"<|im_start|>system\n$instruction<|im_end|>\n" +
      s"<|im_start|>user\n${annotation.result}<|im_end|>\n" +
      "<|im_start|>assistant\n<|endoftext|>"
  }

  private def buildImagePrompt: String = {
    s"<|im_start|>system\n$instruction<|im_end|>\n" +
      "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n" +
      "<|im_start|>assistant\n<|endoftext|>"
  }

  private def encodeTextPrompt(prompt: String): Array[Int] =
    Array(bosTokenId) ++ tokenizeChunk(prompt)

  private def encodeImagePrompt(prompt: String, imageTokenLength: Int): Array[Int] = {
    val imagePadPattern = raw"<\|image_pad\|>".r
    require(
      imagePadPattern.findFirstIn(prompt).isDefined,
      "Image prompt is missing <|image_pad|>.")

    val promptChunks = imagePadPattern
      .split(prompt)
      .toSeq
      .map(tokenizeChunk)

    val imagePaddingTokens = Array.fill(imageTokenLength)(imageTokenId)
    val combinedChunks =
      promptChunks.map(_.toArray).reduce(_ ++ imagePaddingTokens ++ _)

    Array(bosTokenId) ++ combinedChunks ++ imagePromptFixTokenIds
  }

  private def tokenizeChunk(text: String): Array[Int] = {
    val sentence =
      Sentence(content = text, start = 0, end = math.max(text.length - 1, 0), index = 0)
    tokenizer
      .tokenize(sentence)
      .map(tokenizer.encode)
      .flatMap(_.map(_.pieceId))
  }

  private def leftPad(sequences: Seq[Array[Int]]): (Array[Array[Long]], Array[Array[Long]]) = {
    val maxLength = sequences.map(_.length).max
    val inputIds = sequences.map { sequence =>
      Array.fill(maxLength - sequence.length)(padTokenId.toLong) ++ sequence.map(_.toLong)
    }.toArray
    val attentionMask = sequences.map { sequence =>
      Array.fill(maxLength - sequence.length)(0L) ++ Array.fill(sequence.length)(1L)
    }.toArray
    (inputIds, attentionMask)
  }

  private def buildImageTowerInput(annotation: AnnotationImage): ImageTowerInput = {
    val (pixelValues, gridH, gridW) = preprocessImage(annotation)
    val imagePromptTokenLength = (gridH * gridW) / (spatialMergeSize * spatialMergeSize)

    ImageTowerInput(
      inputIds = encodeImagePrompt(buildImagePrompt, imagePromptTokenLength),
      pixelValues = pixelValues,
      imageGridThw = Array(1L, gridH.toLong, gridW.toLong))
  }

  private def preprocessImage(annotation: AnnotationImage): (Array[Array[Float]], Int, Int) = {
    val (resizedHeight, resizedWidth) =
      smartResize(
        annotation.height,
        annotation.width,
        minPixels = minPixels,
        maxPixels = maxPixels)

    val bufferedImage = ImageIOUtils.byteToBufferedImage(
      bytes = annotation.result,
      w = annotation.width,
      h = annotation.height,
      nChannels = annotation.nChannels)

    val resizedImage = ImageResizeUtils.resizeBufferedImage(
      width = resizedWidth,
      height = resizedHeight,
      resample = preprocessor.resample)(bufferedImage)

    val normalizedImage = ImageResizeUtils.normalizeAndConvertBufferedImage(
      img = resizedImage,
      mean = preprocessor.image_mean,
      std = preprocessor.image_std,
      doNormalize = preprocessor.do_normalize,
      doRescale = preprocessor.do_rescale,
      rescaleFactor = preprocessor.rescale_factor)

    val gridH = resizedHeight / patchSize
    val gridW = resizedWidth / patchSize
    val temporalFrames = Array.fill(temporalPatchSize)(normalizedImage)
    val patchVectors =
      Array.ofDim[Float](gridH * gridW, 3 * temporalPatchSize * patchSize * patchSize)

    var patchIndex = 0
    var row = 0
    while (row < gridH) {
      var col = 0
      while (col < gridW) {
        var featureIndex = 0
        var time = 0
        while (time < temporalFrames.length) {
          var channel = 0
          while (channel < temporalFrames(time).length) {
            var patchRow = 0
            while (patchRow < patchSize) {
              var patchCol = 0
              while (patchCol < patchSize) {
                patchVectors(patchIndex)(featureIndex) = temporalFrames(time)(channel)(
                  row * patchSize + patchRow)(col * patchSize + patchCol)
                featureIndex += 1
                patchCol += 1
              }
              patchRow += 1
            }
            channel += 1
          }
          time += 1
        }
        patchIndex += 1
        col += 1
      }
      row += 1
    }

    (patchVectors, gridH, gridW)
  }

  private case class ImageTowerInput(
      inputIds: Array[Int],
      pixelValues: Array[Array[Float]],
      imageGridThw: Array[Long])
}
