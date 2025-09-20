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

import org.apache.spark.ml.param.{BooleanParam, Param}

trait HasReaderProperties extends HasHTMLReaderProperties {

  protected final val inputCol: Param[String] =
    new Param(this, "inputCol", "the output annotation column")

  /** Overrides annotation column name when transforming */
  final def setInputCol(value: String): this.type = set(inputCol, value)

  /** Gets annotation column name going to generate */
  final def getInputCol: String = $(inputCol)

  val contentPath = new Param[String](this, "contentPath", "Path to the content source")

  def setContentPath(value: String): this.type = set(contentPath, value)

  val contentType = new Param[String](
    this,
    "contentType",
    "Set the content type to load following MIME specification")

  def setContentType(value: String): this.type = set(contentType, value)

  val storeContent = new Param[Boolean](
    this,
    "storeContent",
    "Whether to include the raw file content in the output DataFrame as a separate 'content' column, alongside the structured output.")

  def setStoreContent(value: Boolean): this.type = set(storeContent, value)

  val titleFontSize = new Param[Int](
    this,
    "titleFontSize",
    "Minimum font size threshold used as part of heuristic rules to detect title elements based on formatting (e.g., bold, centered, capitalized).")

  def setTitleFontSize(value: Int): this.type = set(titleFontSize, value)

  val inferTableStructure = new Param[Boolean](
    this,
    "inferTableStructure",
    "Whether to generate an HTML table representation from structured table content. When enabled, a full <table> element is added alongside cell-level elements, based on row and column layout.")

  def setInferTableStructure(value: Boolean): this.type = set(inferTableStructure, value)

  val includePageBreaks = new Param[Boolean](
    this,
    "includePageBreaks",
    "Whether to detect and tag content with page break metadata. In Word documents, this includes manual and section breaks. In Excel files, this includes page breaks based on column boundaries.")

  def setIncludePageBreaks(value: Boolean): this.type = set(includePageBreaks, value)

  val ignoreExceptions: BooleanParam =
    new BooleanParam(this, "ignoreExceptions", "whether to ignore exceptions during processing")

  def setIgnoreExceptions(value: Boolean): this.type = set(ignoreExceptions, value)

  setDefault(
    contentPath -> "",
    contentType -> "text/plain",
    storeContent -> false,
    titleFontSize -> 9,
    inferTableStructure -> false,
    includePageBreaks -> false,
    ignoreExceptions -> true,
    inputCol -> "")

}
