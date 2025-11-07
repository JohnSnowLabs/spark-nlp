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
import com.johnsnowlabs.nlp.annotators.cleaners.util.CleanerHelper.DOUBLE_PARAGRAPH_PATTERN
import org.apache.spark.ml.param.{Param, StringArrayParam}

trait HasTextReaderProperties extends ParamsAndFeaturesWritable {

  val titleLengthSize = new Param[Int](
    this,
    "titleLengthSize",
    "Maximum character length used to determine if a text block qualifies as a title during parsing.")

  /** Set the maximum character length used to determine if a text block qualifies as a title
    * during parsing.
    *
    * @param value
    *   maximum number of characters to treat a block as a title
    * @return
    *   this instance with the updated `titleLengthSize` parameter
    */
  def setTitleLengthSize(value: Int): this.type = set(titleLengthSize, value)

  val groupBrokenParagraphs = new Param[Boolean](
    this,
    "groupBrokenParagraphs",
    "Whether to merge fragmented lines into coherent paragraphs using heuristics based on line length and structure.")

  /** Enable or disable merging of fragmented lines into coherent paragraphs when parsing text.
    * When enabled, heuristics based on line length and structure are used to group lines.
    *
    * @param value
    *   true to group broken paragraphs, false to preserve original line breaks
    * @return
    *   this instance with the updated `groupBrokenParagraphs` parameter
    */
  def setGroupBrokenParagraphs(value: Boolean): this.type = set(groupBrokenParagraphs, value)

  val paragraphSplit = new Param[String](
    this,
    "paragraphSplit",
    "Regex pattern used to detect paragraph boundaries when grouping broken paragraphs.")

  /** Set the regular expression used to detect paragraph boundaries when grouping broken
    * paragraphs.
    *
    * @param value
    *   regex pattern string to detect paragraph boundaries
    * @return
    *   this instance with the updated `paragraphSplit` parameter
    */
  def setParagraphSplit(value: String): this.type = set(paragraphSplit, value)

  val shortLineWordThreshold = new Param[Int](
    this,
    "shortLineWordThreshold",
    "Maximum word count for a line to be considered 'short' during broken paragraph grouping.")

  /** Set the maximum number of words for a line to be considered "short" when grouping broken
    * paragraphs. Short lines often indicate line-wrapping within a paragraph rather than a real
    * paragraph break.
    *
    * @param value
    *   maximum word count for a line to be considered short
    * @return
    *   this instance with the updated `shortLineWordThreshold` parameter
    */
  def setShortLineWordThreshold(value: Int): this.type = set(shortLineWordThreshold, value)

  val maxLineCount = new Param[Int](
    this,
    "maxLineCount",
    "Maximum number of lines to evaluate when estimating paragraph layout characteristics.")

  /** Set the maximum number of lines to evaluate when estimating paragraph layout
    * characteristics. This limits the amount of text inspected for layout heuristics.
    *
    * @param value
    *   maximum number of lines to inspect
    * @return
    *   this instance with the updated `maxLineCount` parameter
    */
  def setMaxLineCount(value: Int): this.type = set(maxLineCount, value)

  val threshold = new Param[Double](
    this,
    "threshold",
    "Threshold ratio of empty lines used to decide between new line-based or broken-paragraph grouping.")

  /** Set the threshold ratio of empty lines used to decide between new line-based or
    * broken-paragraph grouping. Lower values make it easier to choose broken-paragraph grouping.
    *
    * @param value
    *   ratio between 0.0 and 1.0 representing the empty-line threshold
    * @return
    *   this instance with the updated `threshold` parameter
    */
  def setThreshold(value: Double): this.type = set(threshold, value)

  val extractTagAttributes = new StringArrayParam(
    this,
    "extractTagAttributes",
    "Extract attribute values into separate lines when parsing tag-based formats (e.g., HTML or XML).")

  /** Specify which tag attributes should have their values extracted as text when parsing
    * tag-based formats (e.g., HTML or XML).
    *
    * @param attributes
    *   array of attribute names to extract
    * @return
    *   this instance with the updated `extractTagAttributes` parameter
    */
  def setExtractTagAttributes(attributes: Array[String]): this.type =
    set(extractTagAttributes, attributes)

  setDefault(
    titleLengthSize -> 50,
    groupBrokenParagraphs -> false,
    paragraphSplit -> DOUBLE_PARAGRAPH_PATTERN,
    shortLineWordThreshold -> 5,
    maxLineCount -> 2000,
    threshold -> 0.1,
    extractTagAttributes -> Array.empty[String])

}
