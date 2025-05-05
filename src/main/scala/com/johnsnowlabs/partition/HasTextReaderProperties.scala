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
import org.apache.spark.ml.param.Param

trait HasTextReaderProperties extends ParamsAndFeaturesWritable {

  val titleLengthSize = new Param[Int](
    this,
    "titleLengthSize",
    "Maximum character length used to determine if a text block qualifies as a title during parsing.")

  def setTitleLengthSize(value: Int): this.type = set(titleLengthSize, value)

  val groupBrokenParagraphs = new Param[Boolean](
    this,
    "groupBrokenParagraphs",
    "Whether to merge fragmented lines into coherent paragraphs using heuristics based on line length and structure.")

  def setGroupBrokenParagraphs(value: Boolean): this.type = set(groupBrokenParagraphs, value)

  val paragraphSplit = new Param[String](
    this,
    "paragraphSplit",
    "Regex pattern used to detect paragraph boundaries when grouping broken paragraphs.")

  def setParagraphSplit(value: String): this.type = set(paragraphSplit, value)

  val shortLineWordThreshold = new Param[Int](
    this,
    "shortLineWordThreshold",
    "Maximum word count for a line to be considered 'short' during broken paragraph grouping.")

  def setShortLineWordThreshold(value: Int): this.type = set(shortLineWordThreshold, value)

  val maxLineCount = new Param[Int](
    this,
    "maxLineCount",
    "Maximum number of lines to evaluate when estimating paragraph layout characteristics.")

  def setMaxLineCount(value: Int): this.type = set(maxLineCount, value)

  val threshold = new Param[Double](
    this,
    "threshold",
    "Threshold ratio of empty lines used to decide between new line-based or broken-paragraph grouping.")

  def setThreshold(value: Double): this.type = set(threshold, value)

  setDefault(
    titleLengthSize -> 50,
    groupBrokenParagraphs -> false,
    paragraphSplit -> DOUBLE_PARAGRAPH_PATTERN,
    shortLineWordThreshold -> 5,
    maxLineCount -> 2000,
    threshold -> 0.1)

}
