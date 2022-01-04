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

import org.apache.spark.ml.param.{BooleanParam, IntParam}

trait HasTransformerProperties extends ParamsAndFeaturesWritable {

  /** Whether to ignore case in index lookups (Default depends on model)
   *
   * @group param
   */
  val caseSensitive = new BooleanParam(this, "caseSensitive", "Whether to ignore case in index lookups")

  /** Max sentence length to process (Default: `128`)
   *
   * @group param
   * */
  val maxSentenceLength = new IntParam(this, "maxSentenceLength", "Max sentence length to process")


  setDefault(caseSensitive -> false, maxSentenceLength -> 128)

  /** @group getParam */
  def getCaseSensitive: Boolean = $(caseSensitive)

  /** @group setParam */
  def setCaseSensitive(value: Boolean): this.type = set(this.caseSensitive, value)

  /** @group getParam */
  def getMaxSentenceLength: Int = $(maxSentenceLength)

  /** @group setParam */
  def setMaxSentenceLength(value: Int): this.type = {
    val transformerModel = this.getClass.getSimpleName.toUpperCase().replace("EMBEDDINGS", "")
    require(value <= 512, s"$transformerModel models do not support sequences longer than 512 because of trainable positional embeddings")
    require(value >= 1, "The maxSentenceLength must be at least 1")
    set(maxSentenceLength, value)
  }

}
