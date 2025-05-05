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

trait HasExcelReaderProperties extends ParamsAndFeaturesWritable {

  val cellSeparator = new Param[String](
    this,
    "cellSeparator",
    "String used to join cell values in a row when assembling textual output.")

  def setCellSeparator(value: String): this.type = set(cellSeparator, value)

  val appendCells = new Param[Boolean](
    this,
    "appendCells",
    "Whether to append all rows into a single content block instead of creating separate elements per row.")

  def setAppendCells(value: Boolean): this.type = set(appendCells, value)

  setDefault(cellSeparator -> "\t", appendCells -> false)

}
