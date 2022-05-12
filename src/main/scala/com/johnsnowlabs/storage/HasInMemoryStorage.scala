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

trait HasInMemoryStorage extends Params {

  val enableInMemoryStorage: BooleanParam = new BooleanParam(
    this,
    "enableInMemoryStorage",
    "whether to include indexed storage in disk or in memory")

  def setEnableInMemoryStorage(value: Boolean): this.type = set(enableInMemoryStorage, value)

  def getEnableInMemoryStorage: Boolean = $(enableInMemoryStorage)

  setDefault(enableInMemoryStorage, false)

}
