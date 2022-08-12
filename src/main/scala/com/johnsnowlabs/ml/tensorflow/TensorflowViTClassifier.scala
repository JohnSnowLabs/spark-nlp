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

package com.johnsnowlabs.ml.tensorflow

import com.johnsnowlabs.ml.tensorflow.sign.{ModelSignatureConstants, ModelSignatureManager}
import com.johnsnowlabs.nlp.annotators.cv.feature_extractor.Preprocessor
import com.johnsnowlabs.nlp.annotators.cv.util.io.ImageIOUtils
import com.johnsnowlabs.nlp.annotators.cv.util.transform.ImageResizeUtils
import com.johnsnowlabs.nlp._

import scala.collection.JavaConverters._

class TensorflowViTClassifier(
    val tensorflowWrapper: TensorflowWrapper,
    configProtoBytes: Option[Array[Byte]] = None,
    tags: Map[String, BigInt],
    imageMean: Array[Double],
    imageStd: Array[Double],
    resample: Int,
    size: Int,
    signatures: Option[Map[String, String]] = None)
    extends Serializable {

  val _tfViTSignatures: Map[String, String] =
    signatures.getOrElse(ModelSignatureManager.apply())

  private def sessionWarmup(): Unit = {
    val image =
      ImageIOUtils.loadImage(getClass.getResourceAsStream("/image/ox.JPEG"))
    val bytes = ImageIOUtils.bufferedImageToByte(image)
    val preprocessor =
      Preprocessor(
        do_normalize = true,
        do_resize = true,
        "ViTFeatureExtractor",
        imageMean,
        imageStd,
        resample,
        size)
    val images =
      Array(AnnotationImage("image", "ox.JPEG", 265, 360, 3, 16, bytes, Map("image" -> "0")))
    val encoded = encode(images, preprocessor)
    tag(encoded)
  }

  sessionWarmup()

  def tag(
      batch: Array[Array[Array[Array[Float]]]],
      activation: String = ActivationFunction.softmax): Array[Array[Float]] = {
    val tensors = new TensorResources()
    val batchLength = batch.length

    val imageTensors = tensors.createTensor(batch)

    val runner = tensorflowWrapper
      .getTFSessionWithSignature(configProtoBytes = configProtoBytes, initAllTables = false)
      .runner

    runner
      .feed(
        _tfViTSignatures
          .getOrElse(ModelSignatureConstants.PixelValuesInput.key, "missing_pixel_values"),
        imageTensors)
      .fetch(_tfViTSignatures
        .getOrElse(ModelSignatureConstants.LogitsOutput.key, "missing_logits_key"))

    val outs = runner.run().asScala
    val rawScores = TensorResources.extractFloats(outs.head)

    tensors.clearSession(outs)
    tensors.clearTensors()
    imageTensors.close()

    val dim = rawScores.length / batchLength
    val batchScores: Array[Array[Float]] =
      rawScores
        .grouped(dim)
        .map(scores => calculateSoftmax(scores))
        .toArray
    batchScores
  }

  /** Calculate softmax from returned logits
    * @param scores
    *   logits output from output layer
    * @return
    */
  def calculateSoftmax(scores: Array[Float]): Array[Float] = {
    val exp = scores.map(x => math.exp(x))
    exp.map(x => x / exp.sum).map(_.toFloat)
  }

  /** Calculate sigmoid from returned logits
    * @param scores
    *   logits output from output layer
    * @return
    */
  def calculateSigmoid(scores: Array[Float]): Array[Float] = {
    scores.map(x => 1 / (1 + Math.exp(-x)).toFloat)
  }

  def predict(
      images: Array[AnnotationImage],
      batchSize: Int,
      preprocessor: Preprocessor,
      activation: String = ActivationFunction.softmax): Seq[Annotation] = {

    images
      .grouped(batchSize)
      .flatMap { batch =>
        val encoded = encode(batch, preprocessor)
        val logits = tag(encoded, activation)

        batch.zip(logits).map { case (image, score) =>
          val label =
            tags.find(_._2 == score.zipWithIndex.maxBy(_._1)._2).map(_._1).getOrElse("NA")

          val meta = score.zipWithIndex.flatMap(x =>
            Map(tags.take(10).find(_._2 == x._2).map(_._1).toString -> x._1.toString))

          val imageMeta = Map(
            "height" -> image.height.toString,
            "width" -> image.width.toString,
            "nChannels" -> image.nChannels.toString,
            "mode" -> image.mode.toString,
            "origin" -> image.origin)

          Annotation(
            annotatorType = AnnotatorType.CATEGORY,
            begin = 0,
            end = label.length - 1,
            result = label,
            metadata = Map("image" -> "0") ++ imageMeta ++ meta)
        }

      }
  }.toSeq

  def encode(
      annotations: Array[AnnotationImage],
      preprocessor: Preprocessor): Array[Array[Array[Array[Float]]]] = {

    val batchProcessedImages = annotations.map { annot =>
      val bufferedImage = ImageIOUtils.byteToBufferedImage(
        bytes = annot.result,
        w = annot.width,
        h = annot.height,
        nChannels = annot.nChannels)

      val resizedImage =
        ImageResizeUtils.resizeBufferedImage(
          width = preprocessor.size,
          height = preprocessor.size,
          Some(annot.nChannels))(bufferedImage)

      ImageResizeUtils.normalizeBufferedImage(
        img = resizedImage,
        mean = preprocessor.image_mean,
        std = preprocessor.image_std)
    }

    batchProcessedImages

  }

}
