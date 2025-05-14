/*
 * Copyright 2017-2023 John Snow Labs
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
import com.johnsnowlabs.ml.onnx.{OnnxSession, OnnxWrapper, TensorResources}
import com.johnsnowlabs.ml.openvino.OpenvinoWrapper
import com.johnsnowlabs.ml.tensorflow.TensorflowWrapper
import com.johnsnowlabs.ml.util.LinAlg.{argmax, softmax}
import com.johnsnowlabs.ml.util.{ONNX, Openvino, TensorFlow}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common.Sentence
import com.johnsnowlabs.nlp.annotators.cv.feature_extractor.Preprocessor
import com.johnsnowlabs.nlp.annotators.cv.util.io.ImageIOUtils
import com.johnsnowlabs.nlp.annotators.cv.util.transform.ImageResizeUtils
import com.johnsnowlabs.nlp.annotators.tokenizer.bpe.CLIPTokenizer

import scala.jdk.CollectionConverters.mapAsJavaMapConverter

private[johnsnowlabs] class CLIP(
    val tensorflowWrapper: Option[TensorflowWrapper],
    val onnxWrapper: Option[OnnxWrapper],
    val openvinoWrapper: Option[OpenvinoWrapper],
    configProtoBytes: Option[Array[Byte]] = None,
    tokenizer: CLIPTokenizer,
    preprocessor: Preprocessor)
    extends Serializable {

  val detectedEngine: String =
    if (tensorflowWrapper.isDefined) TensorFlow.name
    else if (openvinoWrapper.isDefined) Openvino.name
    else if (onnxWrapper.isDefined) ONNX.name
    else throw new IllegalArgumentException("No model engine defined.")

  private val onnxSessionOptions: Map[String, String] = new OnnxSession().getSessionOptions
  private def sessionWarmup(): Unit = {
    val image =
      ImageIOUtils.loadImage(getClass.getResourceAsStream("/image/ox.JPEG"))
    val bytes = ImageIOUtils.bufferedImageToByte(image.get)
    val images =
      Array(AnnotationImage("image", "ox.JPEG", 265, 360, 3, 16, bytes, Map("image" -> "0")))
    predict(images, Array("a photo of an ox"), 1)
  }

  sessionWarmup()

  /* Tags images and labels them */
  def tag(
      batchImages: Array[Array[Array[Array[Float]]]],
      labels: Array[Array[Long]]): Array[Array[Float]] = {

    detectedEngine match {
      case ONNX.name =>
        val (runner, _) = onnxWrapper.get.getSession(onnxSessionOptions)
        val onnxTensorResources = new TensorResources()

        val tokenTensors = onnxTensorResources.createTensor(labels)
        val pixelValuesTensor = onnxTensorResources.createTensor(batchImages)
        val attentionMaskTensor =
          onnxTensorResources.createTensor(Array.fill(labels.length, labels.head.length)(1L))

        val inputs =
          Map(
            "input_ids" -> tokenTensors,
            "pixel_values" -> pixelValuesTensor,
            "attention_mask" -> attentionMaskTensor).asJava

        val results = runner.run(inputs)
        val rawLogits = results
          .get("logits_per_text")
          .get()
          .asInstanceOf[OnnxTensor]
          .getFloatBuffer
          .array()

        val batchSize = batchImages.length

        results.close()
        onnxTensorResources.clearTensors()

        // Original Model Output: (num_labels, batch_size)
        // Transpose to get (batch_size, num_labels)
        val logits = rawLogits.grouped(batchSize).toArray.transpose

        logits.map(scores => softmax(scores))

      case Openvino.name =>
        val tokenTensors =
          new org.intel.openvino.Tensor(Array(labels.length, labels.head.length), labels.flatten)
        val pixelValuesTensor = new org.intel.openvino.Tensor(
          Array(
            batchImages.length,
            batchImages.head.length,
            batchImages.head.head.length,
            batchImages.head.head.head.length),
          batchImages.flatten.flatten.flatten)
        val attentionMaskTensor =
          new org.intel.openvino.Tensor(
            Array(labels.length, labels.head.length),
            Array.fill(labels.length, labels.head.length)(1L).flatten)

        val inferRequest = openvinoWrapper.get.getCompiledModel().create_infer_request()
        inferRequest.set_tensor("input_ids", tokenTensors)
        inferRequest.set_tensor("pixel_values", pixelValuesTensor)
        inferRequest.set_tensor("attention_mask", attentionMaskTensor)
        inferRequest.infer()

        val result = inferRequest.get_tensor("logits_per_text")
        val rawLogits = result.data()

        val batchSize = batchImages.length
        val logits = rawLogits.grouped(batchSize).toArray.transpose

        logits.map(scores => softmax(scores))

      case _ => throw new Exception("Only ONNX is currently supported.")
    }
  }

  def processImage(batch: Array[AnnotationImage]): Array[Array[Array[Array[Float]]]] = {
    batch.map { annot =>
      val bufferedImage = ImageIOUtils.byteToBufferedImage(
        bytes = annot.result,
        w = annot.width,
        h = annot.height,
        nChannels = annot.nChannels)

      val resizedAndCroppedImage =
        ImageResizeUtils.resizeAndCenterCropImage(
          bufferedImage,
          requestedSize = preprocessor.size,
          cropPct = 1,
          resample = preprocessor.resample)

      val normalizedImage = ImageResizeUtils.normalizeAndConvertBufferedImage(
        img = resizedAndCroppedImage,
        mean = preprocessor.image_mean,
        std = preprocessor.image_std,
        doNormalize = preprocessor.do_normalize,
        doRescale = preprocessor.do_rescale,
        rescaleFactor = preprocessor.rescale_factor)

      normalizedImage
    }
  }

  def encodeLabels(labels: Array[String]): Array[Array[Long]] = {
    val tokenIds = labels.map { text =>
      val tokens = tokenizer.tokenize(Sentence(text, 0, text.length, 0))
      tokenizer.encode(tokens).map(_.pieceId.toLong)
    }

    // Pad to same length
    val padToken = tokenizer.specialTokens.pad.id.toLong
    val maxLength = tokenIds.map(_.length).max
    tokenIds.map { tokens =>
      tokens ++ Array.fill(maxLength - tokens.length)(padToken)
    }
  }

  def predict(
      images: Array[AnnotationImage],
      labels: Array[String],
      batchSize: Int): Seq[Annotation] = {

    images
      .grouped(batchSize)
      .flatMap { batch =>
        val processedImages = processImage(batch)
        val encodedLabels = encodeLabels(labels)
        val logits = tag(processedImages, encodedLabels)

        batch.zip(logits).map { case (image, scores) =>
          val maxIndex = argmax(scores)
          val label: String = labels(maxIndex)

          val imageMeta = Map(
            "height" -> image.height.toString,
            "width" -> image.width.toString,
            "nChannels" -> image.nChannels.toString,
            "mode" -> image.mode.toString,
            "origin" -> image.origin)

          val scoreMeta: Map[String, String] = labels.zip(scores.map(_.toString)).toMap

          Annotation(
            annotatorType = AnnotatorType.CATEGORY,
            begin = 0,
            end = label.length - 1,
            result = label,
            metadata = Map("image" -> "0") ++ imageMeta ++ scoreMeta)
        }

      }
  }.toSeq

}
