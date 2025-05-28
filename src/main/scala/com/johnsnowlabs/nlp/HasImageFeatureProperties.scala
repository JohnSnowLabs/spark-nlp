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

package com.johnsnowlabs.nlp

import org.apache.spark.ml.param.{BooleanParam, DoubleArrayParam, IntParam, Param}

/** example of required parameters
  * {{{
  * {
  * "do_normalize": true,
  * "do_resize": true,
  * "feature_extractor_type": "ViTFeatureExtractor",
  * "image_mean": [
  * 0.5,
  * 0.5,
  * 0.5
  * ],
  * "image_std": [
  * 0.5,
  * 0.5,
  * 0.5
  * ],
  * "resample": 2,
  * "size": 224
  * }
  * }}}
  */
trait HasImageFeatureProperties extends ParamsAndFeaturesWritable {

  /** Whether or not to normalize the input with mean and standard deviation
    * @group param
    */
  val doNormalize = new BooleanParam(
    this,
    "doNormalize",
    "Whether to normalize the input with mean and standard deviation")

  /** Whether to resize the input to a certain size
    * @group param
    */
  val doResize =
    new BooleanParam(this, "doResize", "Whether to resize the input to a certain size")

  /** Name of model's architecture for feature extraction
    * @group param
    */
  val featureExtractorType = new Param[String](
    this,
    "featureExtractorType",
    "Name of model's architecture for feature extraction")

  /** The sequence of means for each channel, to be used when normalizing images
    * @group param
    */
  val imageMean = new DoubleArrayParam(
    this,
    "imageMean",
    "The sequence of means for each channel, to be used when normalizing images")

  /** The sequence of standard deviations for each channel, to be used when normalizing images
    * @group param
    */
  val imageStd = new DoubleArrayParam(
    this,
    "imageStd",
    "The sequence of standard deviations for each channel, to be used when normalizing images")

  /** An optional resampling filter. This can be one of PIL.Image.NEAREST, PIL.Image.BOX,
    * PIL.Image.BILINEAR, PIL.Image.HAMMING, PIL.Image.BICUBIC or PIL.Image.LANCZOS. Only has an
    * effect if do_resize is set to True
    * @group param
    */
  val resample = new IntParam(
    this,
    "resample",
    "An optional resampling filter. This can be one of PIL.Image.NEAREST, PIL.Image.BOX, PIL.Image.BILINEAR, " +
      "PIL.Image.HAMMING, PIL.Image.BICUBIC or PIL.Image.LANCZOS. Only has an effect if do_resize is set to True")

  /** Resize the input to the given size. If a tuple is provided, it should be (width, height). If
    * only an integer is provided, then the input will be resized to (size, size). Only has an
    * effect if do_resize is set to True.
    * @group param
    */
  val size = new IntParam(
    this,
    "size",
    "Resize the input to the given size. If a tuple is provided, it should be (width, height). " +
      "If only an integer is provided, then the input will be resized to (size, size). Only has an effect if do_resize is set to True.")

  setDefault(
    doResize -> true,
    doNormalize -> true,
    featureExtractorType -> "ViTFeatureExtractor",
    imageMean -> Array(0.5d, 0.5d, 0.5d),
    imageStd -> Array(0.5d, 0.5d, 0.5d),
    resample -> 2,
    size -> 224)

  /** @group setParam */
  def setDoResize(value: Boolean): this.type = set(this.doResize, value)

  /** @group setParam */
  def setDoNormalize(value: Boolean): this.type = set(this.doNormalize, value)

  /** @group setParam */
  def setFeatureExtractorType(value: String): this.type = set(this.featureExtractorType, value)

  /** @group setParam */
  def setImageMean(value: Array[Double]): this.type =
    set(this.imageMean, value)

  /** @group setParam */
  def setImageStd(value: Array[Double]): this.type =
    set(this.imageStd, value)

  /** @group setParam */
  def setResample(value: Int): this.type = set(this.resample, value)

  /** @group setParam */
  def setSize(value: Int): this.type = set(this.size, value)

  /** @group getParam */
  def getDoResize: Boolean = $(doResize)

  /** @group getParam */
  def getDoNormalize: Boolean = $(doNormalize)

  /** @group getParam */
  def getFeatureExtractorType: String = $(featureExtractorType)

  /** @group getParam */
  def getImageMean: Array[Double] = $(imageMean)

  /** @group getParam */
  def getImageStd: Array[Double] = $(imageStd)

  /** @group getParam */
  def getResample: Int = $(resample) match {
    // Match to AffineTransformOp
    case 0 => 1 // NEAREST
    case 2 => 2 // BILINEAR
    case 3 => 3 // BICUBIC
    case otherValue: Int =>
      throw new IllegalArgumentException(
        s"Invalid value for resampling ($otherValue). Only Nearest Neighbour (0), Bilinear (2) or Bicubic (3) filters are currently supported.")
  }

  /** @group getParam */
  def getSize: Int = $(size)

}
