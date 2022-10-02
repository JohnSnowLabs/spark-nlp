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

import org.apache.spark.ml.param.{BooleanParam, DoubleArrayParam, FloatParam, IntParam, Param}

/** example of required parameters
  * {{{
  * {
  * "do_normalize": true,
  * "feature_size": 1,
  * "padding_side": "right",
  * "padding_value": 0.0,
  * "return_attention_mask": false,
  * "sampling_rate": 16000
  * }
  * }}}
  */
trait HasAudioFeatureProperties extends ParamsAndFeaturesWritable {

  /** Whether or not to normalize the input with mean and standard deviation
    * @group param
    */
  val doNormalize = new BooleanParam(
    this,
    "doNormalize",
    "Whether to normalize the input with mean and standard deviation")

  /** @group param */
  val returnAttentionMask =
    new BooleanParam(this, "returnAttentionMask", "")

  /** @group param */
  val paddingSide = new Param[String](this, "paddingSide", "")

  /** @group param */
  val paddingValue = new FloatParam(this, "paddingValue", "")

  /** @group param */
  val featureSize = new IntParam(this, "featureSize", "")

  /** @group param */
  val samplingRate = new IntParam(this, "samplingRate", "")

  setDefault(
    doNormalize -> true,
    returnAttentionMask -> false,
    paddingSide -> "right",
    paddingValue -> 0.0f,
    featureSize -> 1,
    samplingRate -> 16000)

  /** @group setParam */
  def setDoNormalize(value: Boolean): this.type = set(this.doNormalize, value)

  /** @group setParam */
  def setReturnAttentionMask(value: Boolean): this.type = set(this.returnAttentionMask, value)

  /** @group setParam */
  def setPaddingSide(value: String): this.type = set(this.paddingSide, value)

  /** @group setParam */
  def setPaddingValue(value: Float): this.type = set(this.paddingValue, value)

  /** @group setParam */
  def setFeatureSize(value: Int): this.type = set(this.featureSize, value)

  /** @group setParam */
  def setSamplingRate(value: Int): this.type = set(this.samplingRate, value)

  /** @group getParam */
  def getDoNormalize: Boolean = $(doNormalize)

  /** @group getParam */
  def getReturnAttentionMask: Boolean = $(returnAttentionMask)

  /** @group getParam */
  def getPaddingSide: String = $(paddingSide)

  /** @group getParam */
  def getPaddingValue: Float = $(paddingValue)

  /** @group getParam */
  def getFeatureSize: Int = $(featureSize)

  /** @group getParam */
  def getSamplingRate: Int = $(samplingRate)

}
