/*
 * Copyright 2017-2025 John Snow Labs
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
package com.johnsnowlabs.partition

import com.johnsnowlabs.nlp.ParamsAndFeaturesWritable
import org.apache.spark.ml.param.Param

trait HasXmlReaderProperties extends ParamsAndFeaturesWritable {

  val xmlKeepTags = new Param[Boolean](
    this,
    "xmlKeepTags",
    "Whether to include XML tag names as metadata in the output.")

  def setXmlKeepTags(value: Boolean): this.type = set(xmlKeepTags, value)

  val onlyLeafNodes = new Param[Boolean](
    this,
    "onlyLeafNodes",
    "If true, only processes XML leaf nodes (no nested children).")

  def setOnlyLeafNodes(value: Boolean): this.type = set(onlyLeafNodes, value)

  setDefault(xmlKeepTags -> false, onlyLeafNodes -> true)
}
