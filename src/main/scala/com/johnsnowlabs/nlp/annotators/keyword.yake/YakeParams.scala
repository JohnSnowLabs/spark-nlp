/*
 * Copyright 2017-2021 John Snow Labs
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

package com.johnsnowlabs.nlp.annotators.keyword.yake

import org.apache.spark.ml.param.{BooleanParam, FloatParam, IntParam, Params, StringArrayParam}

trait YakeParams extends Params {


  /** Window size for Co-Occurrence (Default: `3`).
   * Yake will construct a co-occurrence matrix. You can set the window size for the co-occurrence matrix construction
   * with this parameter.
   * Example: `windowSize=2` will look at two words to both left and right of a candidate word.
   *
   * @group param
   */
  val windowSize = new IntParam(this, "windowSize", "Window size for Co-Occurrence")

  /** Maximum N-grams a keyword should have (Default: `3`).
   *
   * @group param
   */
  val maxNGrams = new IntParam(this, "maxNGrams", "Maximum N-grams a keyword should have")

  /** Minimum N-grams a keyword should have (Default: `1`).
   *
   * @group param
   */
  val minNGrams = new IntParam(this, "minNGrams", "Minimum N-grams a keyword should have")

  /** Number of Keywords to extract (Default: `30`).
   *
   * @group param
   */
  val nKeywords = new IntParam(this, "nKeywords", "Number of Keywords to extract")

  /** Threshold to filter keywords (Default: `-1`). By default it is disabled.
   * Each keyword will be given a keyword score greater than 0. (The lower the score better the keyword).
   * This sets the upper bound for the keyword score.
   *
   * @group param
   */
  val threshold = new FloatParam(this, "threshold", "Threshold to filter keywords")

  /** the words to be filtered out (Default: English stop words from MLlib)
   *
   * @group param
   */
  val stopWords: StringArrayParam = {
    new StringArrayParam(this, "stopWords", "the words to be filtered out. by default it's english stop words from Spark ML")
  }

  /** @group setParam */
  def setStopWords(value: Array[String]): this.type = set(stopWords, value)
  /** @group getParam */
  def getStopWords: Array[String] = $(stopWords)

  /** @group setParam */
  def setWindowSize(value: Int): this.type = set(windowSize, value+1)
  /** @group setParam */
  def setMaxNGrams(value: Int): this.type = set(maxNGrams, value)
  /** @group setParam */
  def setMinNGrams(value: Int): this.type = set(minNGrams, value)
  /** @group setParam */
  def setNKeywords(value: Int): this.type = set(nKeywords, value)
  /** @group setParam */
  def setThreshold(value: Float): this.type = set(threshold,value)
}
