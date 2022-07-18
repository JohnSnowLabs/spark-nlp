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

package com.johnsnowlabs.nlp.annotators.sbd

import org.apache.spark.ml.param.{BooleanParam, IntParam, Param, Params, StringArrayParam}

import scala.collection.mutable.ArrayBuffer

/** See
  * [[https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/nlp/annotators/sbd/pragmatic]]
  * for further reference on how to use this API
  */
trait SentenceDetectorParams extends Params {

  /** Whether to apply abbreviations at sentence detection (Default: `true`)
    *
    * @group param
    */
  val useAbbrevations = new BooleanParam(
    this,
    "useAbbreviations",
    "whether to apply abbreviations at sentence detection")

  /** Whether take lists into consideration at sentence detection (Default: `true`)
    *
    * @group param
    */
  val detectLists = new BooleanParam(
    this,
    "detectLists",
    "whether take lists into consideration at sentence detection")

  /** Whether to only utilize custom bounds for sentence detection (Default: `false`)
    *
    * @group param
    */
  val useCustomBoundsOnly = new BooleanParam(
    this,
    "useCustomBoundsOnly",
    "whether to only utilize custom bounds for sentence detection")

  /** Whether to explode each sentence into a different row, for better parallelization (Default:
    * `false`)
    *
    * @group param
    */
  val explodeSentences = new BooleanParam(
    this,
    "explodeSentences",
    "whether to explode each sentence into a different row, for better parallelization. Defaults to false.")

  /** Characters used to explicitly mark sentence bounds (Default: None)
    *
    * @group param
    */
  val customBounds: StringArrayParam = new StringArrayParam(
    this,
    "customBounds",
    "characters used to explicitly mark sentence bounds")

  /** Length at which sentences will be forcibly split (Ignored if not set)
    *
    * @group param
    */
  val splitLength: IntParam =
    new IntParam(this, "splitLength", "length at which sentences will be forcibly split.")

  /** Set the minimum allowed length for each sentence (Default: `0`)
    *
    * @group param
    */
  val minLength =
    new IntParam(this, "minLength", "Set the minimum allowed length for each sentence")

  /** Set the maximum allowed length for each sentence (Ignored if not set)
    *
    * @group param
    */
  val maxLength =
    new IntParam(this, "maxLength", "Set the maximum allowed length for each sentence")

  /** How to return matched custom bounds (Default: `none`). Will have no effect if no custom
    * bounds are used. Possible values are:
    *
    *   - "none" - Will not return the matched bound
    *   - "prepend" - Prepends a sentence break to the match
    *   - "append" - Appends a sentence break to the match
    *
    * @group param
    */
  val customBoundsStrategy: Param[String] =
    new Param[String](this, "customBoundsStrategy", "How to return matched custom bounds")

  setDefault(
    useAbbrevations -> true,
    detectLists -> true,
    useCustomBoundsOnly -> false,
    explodeSentences -> false,
    customBounds -> Array.empty[String],
    minLength -> 0,
    customBoundsStrategy -> "none")

  /** Custom sentence separator text
    * @group setParam
    */
  def setCustomBounds(value: Array[String]): this.type = set(customBounds, value)

  /** Custom sentence separator text
    * @group getParam
    */
  def getCustomBounds: Array[String] = $(customBounds)

  /** Use only custom bounds without considering those of Pragmatic Segmenter. Defaults to false.
    * Needs customBounds.
    * @group setParam
    */
  def setUseCustomBoundsOnly(value: Boolean): this.type = set(useCustomBoundsOnly, value)

  /** Use only custom bounds without considering those of Pragmatic Segmenter. Defaults to false.
    * Needs customBounds.
    * @group getParam
    */
  def getUseCustomBoundsOnly: Boolean = $(useCustomBoundsOnly)

  /** Sets how to return matched custom bounds (Default: `none`). Will have no effect if no custom
    * bounds are used. Possible values are:
    *
    *   - "none" - Will not return the matched bound
    *   - "prepend" - Prepends a sentence break to the match
    *   - "append" - Appends a sentence break to the match
    *
    * @group setParam
    */
  def setCustomBoundsStrategy(value: String): this.type = {
    val possibleValues = Array("none", "prepend", "append")
    require(
      possibleValues.contains(value),
      s"$value is not a valid strategy for custom bounds. " +
        s"Possible Values: (${possibleValues.mkString(", ")}).")

    set(customBoundsStrategy, value)
  }

  /** Gets how to return matched custom bounds (Default: `none`).
    * @group getParam
    */
  def getCustomBoundsStrategy: String = $(customBoundsStrategy)

  /** Whether to consider abbreviation strategies for better accuracy but slower performance.
    * Defaults to true.
    * @group setParam
    */
  def setUseAbbreviations(value: Boolean): this.type = set(useAbbrevations, value)

  /** Whether to consider abbreviation strategies for better accuracy but slower performance.
    * Defaults to true.
    * @group getParam
    */
  def getUseAbbreviations: Boolean = $(useAbbrevations)

  /** Whether to take lists into consideration at sentence detection. Defaults to true.
    * @group setParam
    */
  def setDetectLists(value: Boolean): this.type = set(detectLists, value)

  /** Whether to take lists into consideration at sentence detection. Defaults to true.
    * @group getParam
    */
  def getDetectLists: Boolean = $(detectLists)

  /** Whether to split sentences into different Dataset rows. Useful for higher parallelism in fat
    * rows. Defaults to false.
    * @group setParam
    */
  def setExplodeSentences(value: Boolean): this.type = set(explodeSentences, value)

  /** Whether to split sentences into different Dataset rows. Useful for higher parallelism in fat
    * rows. Defaults to false.
    * @group getParam
    */
  def getExplodeSentences: Boolean = $(explodeSentences)

  /** Length at which sentences will be forcibly split
    * @group setParam
    */
  def setSplitLength(value: Int): this.type = set(splitLength, value)

  /** Length at which sentences will be forcibly split
    * @group getParam
    */
  def getSplitLength: Int = $(splitLength)

  /** Set the minimum allowed length for each sentence
    * @group setParam
    */
  def setMinLength(value: Int): this.type = {
    require(value >= 0, "minLength must be greater equal than 0")
    require(value.isValidInt, "minLength must be Int")
    set(minLength, value)
  }

  /** Get the minimum allowed length for each sentence
    * @group getParam
    */
  def getMinLength(value: Int): Int = $(minLength)

  /** Set the maximum allowed length for each sentence
    * @group setParam
    */
  def setMaxLength(value: Int): this.type = {
    require(
      value >= $ {
        minLength
      },
      "maxLength must be greater equal than minLength")
    require(value.isValidInt, "minLength must be Int")
    set(maxLength, value)
  }

  /** Get the maximum allowed length for each sentence
    * @group getParam
    */
  def getMaxLength(value: Int): Int = $(maxLength)

  def truncateSentence(sentence: String, maxLength: Int): Array[String] = {
    var currentLength = 0
    val allSentences = ArrayBuffer.empty[String]
    val currentSentence = ArrayBuffer.empty[String]

    def addWordToSentence(word: String): Unit = {

      /** Adds +1 because of the space joining words */
      currentLength += word.length + 1
      currentSentence.append(word)
    }

    sentence
      .split(" ")
      .foreach(word => {
        if (currentLength + word.length > maxLength) {
          allSentences.append(currentSentence.mkString(" "))
          currentSentence.clear()
          currentLength = 0
          addWordToSentence(word)
        } else {
          addWordToSentence(word)
        }
      })

    /** add leftovers */
    allSentences.append(currentSentence.mkString(" "))
    allSentences.toArray
  }

}
