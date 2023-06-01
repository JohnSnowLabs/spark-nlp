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

package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorType.TOKEN
import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.nlp.{
  Annotation,
  AnnotatorModel,
  HasSimpleAnnotate,
  ParamsAndFeaturesReadable
}
import org.apache.spark.ml.param.{BooleanParam, IntParam, StringArrayParam}
import org.apache.spark.ml.util.Identifiable

/** Instantiated Model of the [[Normalizer]]. For usage and examples, please see the documentation
  * of that class.
  *
  * @see
  *   [[Normalizer]] for the base class
  * @param uid
  *   required internal uid for saving annotator
  * @groupname anno Annotator types
  * @groupdesc anno
  *   Required input and expected output annotator types
  * @groupname Ungrouped Members
  * @groupname param Parameters
  * @groupname setParam Parameter setters
  * @groupname getParam Parameter getters
  * @groupname Ungrouped Members
  * @groupprio param  1
  * @groupprio anno  2
  * @groupprio Ungrouped 3
  * @groupprio setParam  4
  * @groupprio getParam  5
  * @groupdesc param
  *   A list of (hyper-)parameter keys this annotator can take. Users can set and get the
  *   parameter values through setters and getters, respectively.
  */
class NormalizerModel(override val uid: String)
    extends AnnotatorModel[NormalizerModel]
    with HasSimpleAnnotate[NormalizerModel] {

  /** Output annotator type : TOKEN
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = TOKEN

  /** Input annotator type : TOKEN
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN)

  case class TokenizerAndNormalizerMap(
      beginTokenizer: Int,
      endTokenizer: Int,
      token: String,
      beginNormalizer: Int,
      endNormalizer: Int,
      normalizer: String)

  /** normalization regex patterns which match will be removed from token
    *
    * @group param
    */
  val cleanupPatterns = new StringArrayParam(
    this,
    "cleanupPatterns",
    "normalization regex patterns which match will be removed from token")

  /** WARNING: this is for internal use and not intended for users
   * @group setParam */
  def setCleanupPatterns(value: Array[String]): this.type = set(cleanupPatterns, value)

  /** WARNING: this is for internal use and not intended for users
   * @group setParam */
  def getCleanupPatterns: Array[String] = $(cleanupPatterns)

  /** whether to convert strings to lowercase
    *
    * @group param
    */
  val lowercase = new BooleanParam(this, "lowercase", "whether to convert strings to lowercase")

  /** WARNING: this is for internal use and not intended for users
   * @group setParam */
  def setLowercase(value: Boolean): this.type = set(lowercase, value)

  /** WARNING: this is for internal use and not intended for users
   * @group setParam */
  def getLowercase: Boolean = $(lowercase)

  /** slangDict
    * WARNING: this is for internal use and not intended for users
    * @group param
    */
  protected val slangDict: MapFeature[String, String] = new MapFeature(this, "slangDict")

  /** whether or not to be case sensitive to match slangs. Defaults to false.
    *
    * @group param
    */
  val slangMatchCase = new BooleanParam(
    this,
    "slangMatchCase",
    "whether or not to be case sensitive to match slangs. Defaults to false.")

  /** WARNING: this is for internal use and not intended for users
   * @group setParam */
  def setSlangMatchCase(value: Boolean): this.type = set(slangMatchCase, value)

  /** WARNING: this is for internal use and not intended for users
   * @group getParam */
  def getSlangMatchCase: Boolean = $(slangMatchCase)

  def this() = this(Identifiable.randomUID("NORMALIZER"))

  /** WARNING: this is for internal use and not intended for users
   * @group setParam */
  def setSlangDict(value: Map[String, String]): this.type = set(slangDict, value)

  /** Set the minimum allowed length for each token
    * WARNING: this is for internal use and not intended for users
    * @group param
    */
  val minLength = new IntParam(this, "minLength", "Set the minimum allowed length for each token")

  /** WARNING: this is for internal use and not intended for users
   * @group setParam */
  def setMinLength(value: Int): this.type = {
    require(value >= 0, "minLength must be greater equal than 0")
    require(value.isValidInt, "minLength must be Int")
    set(minLength, value)
  }

  /** WARNING: this is for internal use and not intended for users
   * @group getParam */
  def getMinLength: Int = $(minLength)

  /** Set the maximum allowed length for each token
    * WARNING: this is for internal use and not intended for users
    * @group param
    */
  val maxLength = new IntParam(this, "maxLength", "Set the maximum allowed length for each token")

  /** WARNING: this is for internal use and not intended for users
   * @group setParam */
  def setMaxLength(value: Int): this.type = {
    require(
      value >= $ {
        minLength
      },
      "maxLength must be greater equal than minLength")
    require(value.isValidInt, "minLength must be Int")
    set(maxLength, value)
  }

  /** WARNING: this is for internal use and not intended for users
   * @group getParam */
  def getMaxLength: Int = $(maxLength)

  def applyRegexPatterns(word: String): String = {

    val nToken = {
      get(cleanupPatterns)
        .map(_.foldLeft(word)((currentText, compositeToken) => {
          currentText.replaceAll(compositeToken, "")
        }))
        .getOrElse(word)
    }
    nToken
  }

  /** Txt file with delimited words to be transformed into something else
    * WARNING: this is for internal use and not intended for users
    * @group getParam
    */
  protected def getSlangDict: Map[String, String] = $$(slangDict)

  /** ToDo: Review implementation, Current implementation generates spaces between non-words,
    * potentially breaking tokens
    */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val normalizedAnnotations = annotations.flatMap { originalToken =>
      /** slang dictionary keys should have been lowercased if slangMatchCase is false */
      val unslanged = $$(slangDict).get(
        if ($(slangMatchCase)) originalToken.result
        else originalToken.result.toLowerCase)

      /** simple-tokenize the unslanged slag phrase */
      val tokenizedUnslang = {
        unslanged
          .map(unslang => {
            unslang.split(" ")
          })
          .getOrElse(Array(originalToken.result))
      }

      val cleaned = tokenizedUnslang.map(word => applyRegexPatterns(word))

      val cased = if ($(lowercase)) cleaned.map(_.toLowerCase) else cleaned

      cased
        .filter(t =>
          t.nonEmpty && t.length >= $(minLength) && get(maxLength).forall(m => t.length <= m))
        .map { finalToken =>
          {
            Annotation(
              outputAnnotatorType,
              originalToken.begin,
              originalToken.begin + finalToken.length - 1,
              finalToken,
              originalToken.metadata)
          }
        }

    }

    normalizedAnnotations

  }

}

object NormalizerModel extends ParamsAndFeaturesReadable[NormalizerModel]
