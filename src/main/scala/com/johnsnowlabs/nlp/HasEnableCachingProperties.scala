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

import org.apache.spark.ml.param.BooleanParam

trait HasEnableCachingProperties extends ParamsAndFeaturesWritable {

  /** Whether to enable caching DataFrames or RDDs during the training
    * WARNING: this is for internal use and not intended for users
    * @group param
    */
  val enableCaching = new BooleanParam(
    this,
    "enableCaching",
    "Whether to enable caching DataFrames or RDDs during the training")

  setDefault(enableCaching, false)

  /** WARNING: this is for internal use and not intended for users
   * @group getParam */
  def getEnableCaching: Boolean = $(enableCaching)

  /** WARNING: this is for internal use and not intended for users
   * @group setParam */
  def setEnableCaching(value: Boolean): this.type = set(this.enableCaching, value)

}
