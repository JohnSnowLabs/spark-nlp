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

import com.johnsnowlabs.ml.tensorflow.TensorflowWrapper
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.cv.feature_extractor.Preprocessor
import com.johnsnowlabs.nlp.annotators.cv.util.io.ImageIOUtils
import com.johnsnowlabs.nlp.annotators.cv.util.transform.ImageResizeUtils
import com.johnsnowlabs.ml.onnx.OnnxWrapper

private[johnsnowlabs] class ConvNextClassifier(
    tensorflowWrapper: Option[TensorflowWrapper],
    onnxWrapper: Option[OnnxWrapper],
    configProtoBytes: Option[Array[Byte]] = None,
    tags: Map[String, BigInt],
    preprocessor: Preprocessor,
    signatures: Option[Map[String, String]] = None)
    extends ViTClassifier(
      tensorflowWrapper,
      onnxWrapper,
      configProtoBytes,
      tags,
      preprocessor,
      signatures) {

  override def encode(
      annotations: Array[AnnotationImage],
      preprocessor: Preprocessor): Array[Array[Array[Array[Float]]]] = {
    annotations.map { annot =>
      val bufferedImage = ImageIOUtils.byteToBufferedImage(
        bytes = annot.result,
        w = annot.width,
        h = annot.height,
        nChannels = annot.nChannels)

      val resizedImage =
        if (preprocessor.crop_pct.isDefined && preprocessor.size < 384)
          ImageResizeUtils.resizeAndCenterCropImage(
            bufferedImage,
            requestedSize = preprocessor.size,
            cropPct = preprocessor.crop_pct.get,
            resample = preprocessor.resample)
        else
          ImageResizeUtils.resizeBufferedImage(
            width = preprocessor.size,
            height = preprocessor.size,
            resample = preprocessor.resample)(bufferedImage)

      val normalizedImage = ImageResizeUtils.normalizeAndConvertBufferedImage(
        img = resizedImage,
        mean = preprocessor.image_mean,
        std = preprocessor.image_std,
        doNormalize = preprocessor.do_normalize,
        doRescale = preprocessor.do_rescale,
        rescaleFactor = preprocessor.rescale_factor)

      normalizedImage
    }
  }

}
