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
import com.johnsnowlabs.reader.util.pdf.TextStripperType
import org.apache.spark.ml.param.{BooleanParam, IntParam, Param}

trait HasPdfToTextProperties extends ParamsAndFeaturesWritable {

  final val pageNumCol = new Param[String](this, "pageNumCol", "Page number output column name.")
  final val originCol =
    new Param[String](this, "originCol", "Input column name with original path of file.")
  final val partitionNum = new IntParam(this, "partitionNum", "Number of partitions.")
  final val storeSplittedPdf =
    new BooleanParam(this, "storeSplittedPdf", "Force to store bytes content of splitted pdf.")
  final val splitPage = new BooleanParam(
    this,
    "splitPage",
    "Enable/disable splitting per page to identify page numbers and improve performance.")
  final val onlyPageNum = new BooleanParam(this, "onlyPageNum", "Extract only page numbers.")
  final val textStripper = new Param[String](
    this,
    "textStripper",
    "Text stripper type used for output layout and formatting")
  final val sort = new BooleanParam(this, "sort", "Enable/disable sorting content on the page.")
  final val extractCoordinates =
    new BooleanParam(this, "extractCoordinates", "Force extract coordinates of text.")
  final val normalizeLigatures = new BooleanParam(
    this,
    "normalizeLigatures",
    "Whether to convert ligature chars such as 'ﬂ' into its corresponding chars (e.g., {'f', 'l'}).")

  /** @group setParam */
  def setPageNumCol(value: String): this.type = set(pageNumCol, value)

  /** @group getParam */
  def setOriginCol(value: String): this.type = set(originCol, value)

  /** @group getParam */
  def setPartitionNum(value: Int): this.type = set(partitionNum, value)

  /** @group setParam */
  def setStoreSplittedPdf(value: Boolean): this.type = set(storeSplittedPdf, value)

  /** @group setParam */
  def setSplitPage(value: Boolean): this.type = set(splitPage, value)

  /** @group setParam */
  def setOnlyPageNum(value: Boolean): this.type = set(onlyPageNum, value)

  /** @group setParam */
  def setTextStripper(value: String): this.type = set(textStripper, value)

  /** @group setParam */
  def setSort(value: Boolean): this.type = set(sort, value)

  /** @group setParam */
  def setExtractCoordinates(value: Boolean): this.type = set(extractCoordinates, value)

  /** @group setParam */
  def setNormalizeLigatures(value: Boolean): this.type = set(normalizeLigatures, value)

  setDefault(
    pageNumCol -> "pagenum",
    originCol -> "path",
    partitionNum -> 0,
    storeSplittedPdf -> false,
    onlyPageNum -> false,
    splitPage -> true,
    sort -> false,
    textStripper -> TextStripperType.PDF_TEXT_STRIPPER,
    extractCoordinates -> false,
    normalizeLigatures -> true)

}
