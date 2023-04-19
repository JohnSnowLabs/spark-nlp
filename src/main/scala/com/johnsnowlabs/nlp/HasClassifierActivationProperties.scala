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

import org.apache.spark.ml.param.{FloatParam, Param}

trait HasClassifierActivationProperties extends ParamsAndFeaturesWritable {

  /** Whether to enable caching DataFrames or RDDs during the training (Default depends on model).
    *
    * @group param
    */
  val activation: Param[String] = new Param(
    this,
    "activation",
    "Whether to calculate logits via Softmax or Sigmoid. Default is Softmax")

  /** Choose the threshold to determine which logits are considered to be positive or negative.
    * (Default: `0.5f`). The value should be between 0.0 and 1.0. Changing the threshold value
    * will affect the resulting labels and can be used to adjust the balance between precision and
    * recall in the classification process.
    *
    * @group param
    */
  val threshold = new FloatParam(
    this,
    "threshold",
    "Choose the threshold to determine which logits are considered to be positive or negative")

  setDefault(activation -> ActivationFunction.softmax, threshold -> 0.5f)

  /** @group getParam */
  def getActivation: String = $(activation)

  /** @group setParam */
  def setActivation(value: String): this.type = {

    value match {
      case ActivationFunction.softmax =>
        set(this.activation, ActivationFunction.softmax)
      case ActivationFunction.sigmoid =>
        set(this.activation, ActivationFunction.sigmoid)
      case _ =>
        set(this.activation, ActivationFunction.softmax)
    }

  }

  /** Choose the threshold to determine which logits are considered to be positive or negative.
    * (Default: `0.5f`). The value should be between 0.0 and 1.0. Changing the threshold value
    * will affect the resulting labels and can be used to adjust the balance between precision and
    * recall in the classification process.
    *
    * @group param
    */
  def setThreshold(threshold: Float): this.type =
    set(this.threshold, threshold)

}

object ActivationFunction {

  val softmax = "softmax"
  val sigmoid = "sigmoid"

}
