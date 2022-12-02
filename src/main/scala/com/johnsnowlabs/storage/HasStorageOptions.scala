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

package com.johnsnowlabs.storage

import org.apache.spark.ml.param.{BooleanParam, Params}

trait HasStorageOptions extends Params {

  val includeStorage: BooleanParam = new BooleanParam(
    this,
    "includeStorage",
    "whether to include indexed storage in trained model")

  def setIncludeStorage(value: Boolean): this.type = set(includeStorage, value)

  def getIncludeStorage: Boolean = $(includeStorage)

  val enableInMemoryStorage: BooleanParam = new BooleanParam(
    this,
    "enableInMemoryStorage",
    "whether to load whole indexed storage in memory (in-memory lookup)")

  def setEnableInMemoryStorage(value: Boolean): this.type = set(enableInMemoryStorage, value)

  def getEnableInMemoryStorage: Boolean = $(enableInMemoryStorage)

  setDefault(includeStorage -> true, enableInMemoryStorage -> false)

}
