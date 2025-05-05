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
package com.johnsnowlabs.reader.util

import com.johnsnowlabs.nlp.ParamsAndFeaturesWritable
import org.apache.spark.ml.param.{BooleanParam, IntParam, Param}

trait HasPdfProperties extends ParamsAndFeaturesWritable {

  final val pageNumCol = new Param[String](this, "pageNumCol", "Page number output column name.")
  final val originCol =
    new Param[String](this, "originCol", "Input column name with original path of file.")
  final val partitionNum = new IntParam(this, "partitionNum", "Number of partitions.")
  final val storeSplittedPdf =
    new BooleanParam(this, "storeSplittedPdf", "Force to store bytes content of splitted pdf.")

  /** @group setParam */
  def setPageNumCol(value: String): this.type = set(pageNumCol, value)

  /** @group getParam */
  def setOriginCol(value: String): this.type = set(originCol, value)

  /** @group getParam */
  def setPartitionNum(value: Int): this.type = set(partitionNum, value)

  /** @group setParam */
  def setStoreSplittedPdf(value: Boolean): this.type = set(storeSplittedPdf, value)

  setDefault(
    pageNumCol -> "pagenum",
    originCol -> "path",
    partitionNum -> 0,
    storeSplittedPdf -> false)

}
