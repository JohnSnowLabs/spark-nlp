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

import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, HasSimpleAnnotate}
import org.apache.spark.ml.param.{BooleanParam, IntParam, Param, ParamValidators}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

import scala.util.matching.Regex

/** A tokenizer that splits text by a regex pattern.
  *
  * The pattern needs to be set with `setPattern` and this sets the delimiting pattern or how the
  * tokens should be split. By default this pattern is `\s+` which means that tokens should be
  * split by 1 or more whitespace characters.
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotators.RegexTokenizer
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val regexTokenizer = new RegexTokenizer()
  *   .setInputCols("document")
  *   .setOutputCol("regexToken")
  *   .setToLowercase(true)
  *   .setPattern("\\s+")
  *
  * val pipeline = new Pipeline().setStages(Array(
  *     documentAssembler,
  *     regexTokenizer
  *   ))
  *
  * val data = Seq("This is my first sentence.\nThis is my second.").toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.selectExpr("regexToken.result").show(false)
  * +-------------------------------------------------------+
  * |result                                                 |
  * +-------------------------------------------------------+
  * |[this, is, my, first, sentence., this, is, my, second.]|
  * +-------------------------------------------------------+
  * }}}
  *
  * @param uid
  *   required uid for storing annotator to disk
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
class RegexTokenizer(override val uid: String)
    extends AnnotatorModel[RegexTokenizer]
    with HasSimpleAnnotate[RegexTokenizer] {

  import com.johnsnowlabs.nlp.AnnotatorType._

  /** Output annotator type: TOKEN
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = TOKEN

  /** Input annotator type: DOCUMENT
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT)

  def this() = this(Identifiable.randomUID("RegexTokenizer"))

  /** Regex pattern used to match delimiters (Default: `"\\s+"`)
    *
    * @group param
    */
  val pattern: Param[String] = new Param(this, "pattern", "regex pattern used for tokenizing")

  /** @group setParam */
  def setPattern(value: String): this.type = set(pattern, value)

  /** WARNING: this is for internal use and not intended for users
   * @group getParam */
  def getPattern: String = $(pattern)

  /** Indicates whether to convert all characters to lowercase before tokenizing (Default:
    * `false`).
    *
    * @group param
    */
  val toLowercase: BooleanParam = new BooleanParam(
    this,
    "toLowercase",
    "Indicates whether to convert all characters to lowercase before tokenizing.\n")

  /** @group setParam */
  def setToLowercase(value: Boolean): this.type = set(toLowercase, value)

  /** WARNING: this is for internal use and not intended for users
   * @group getParam */
  def getToLowercase: Boolean = $(toLowercase)

  /** Minimum token length, greater than or equal to 0 (Default: `1`). Default is 1, to avoid
    * returning empty strings.
    *
    * @group param
    */
  val minLength: IntParam =
    new IntParam(this, "minLength", "minimum token length (>= 0)", ParamValidators.gtEq(0))

  /** @group setParam */
  def setMinLength(value: Int): this.type = set(minLength, value)

  /** WARNING: this is for internal use and not intended for users
   * @group getParam */
  def getMinLength: Int = $(minLength)

  /** Maximum token length, greater than or equal to 1.
    *
    * @group param
    */
  val maxLength: IntParam =
    new IntParam(this, "maxLength", "maximum token length (>= 1)", ParamValidators.gtEq(1))

  /** @group setParam */
  def setMaxLength(value: Int): this.type = set(maxLength, value)

  /** WARNING: this is for internal use and not intended for users
   * @group getParam */
  def getMaxLength: Int = $(maxLength)

  /** Indicates whether to apply the regex tokenization using a positional mask to guarantee the
    * incremental progression (Default: `false`).
    *
    * @group param
    */
  val positionalMask: BooleanParam =
    new BooleanParam(
      this,
      "positionalMask",
      "Using a positional mask to guarantee the incremental progression of the tokenization.")

  /** @group setParam */
  def setPositionalMask(value: Boolean): this.type = set(positionalMask, value)

  /** WARNING: this is for internal use and not intended for users
   * @group getParam */
  def getPositionalMask: Boolean = $(positionalMask)

  /** Indicates whether to use a trimWhitespace flag to remove whitespaces from identified tokens.
    * (Default: `false`).
    *
    * @group param
    */
  val trimWhitespace: BooleanParam =
    new BooleanParam(
      this,
      "trimWhitespace",
      "Using a trimWhitespace flag to remove whitespaces from identified tokens.")

  /** @group setParam */
  def setTrimWhitespace(value: Boolean): this.type = set(trimWhitespace, value)

  /** WARNING: this is for internal use and not intended for users
   * @group getParam */
  def getTrimWhitespace: Boolean = $(trimWhitespace)

  /** Indicates whether to use a preserve initial indexes before eventual whitespaces removal in
    * tokens. (Default: `false`).
    *
    * @group param
    */
  val preservePosition: BooleanParam =
    new BooleanParam(
      this,
      "preservePosition",
      "Using a preservePosition flag to preserve initial indexes before eventual whitespaces removal in tokens.")

  /** @group setParam */
  def setPreservePosition(value: Boolean): this.type = set(preservePosition, value)

  /** WARNING: this is for internal use and not intended for users
   * @group getParam */
  def getPreservePosition: Boolean = $(preservePosition)

  setDefault(
    inputCols -> Array(DOCUMENT),
    outputCol -> "regexToken",
    toLowercase -> false,
    minLength -> 1,
    pattern -> "\\s+",
    positionalMask -> false,
    trimWhitespace -> false,
    preservePosition -> true)

  /** This func generates a Seq of TokenizedSentences from a Seq of Sentences preserving
    * positional progression
    *
    * @param sentences
    *   to tag
    * @return
    *   Seq of TokenizedSentence objects
    */
  def tagWithPositionalMask(sentences: Seq[Sentence]): Seq[TokenizedSentence] = {

    def calculateIndex(
        indexType: String,
        mask: Array[Int],
        text: String,
        token: String,
        sentenceOffset: Int) = {
      val tokenBeginIndex: Int = text
        .substring(mask.indexOf(0), text.length)
        .indexOf(token) + mask.indexOf(0)
      val index = indexType match {
        case "begin" => tokenBeginIndex
        case "end" =>
          val endIndex = tokenBeginIndex + token.length
          for (i <- Range(0, endIndex)) mask(i) = 1
          if (endIndex == 0) endIndex else endIndex - 1
      }
      index + sentenceOffset
    }

    sentences.map { text =>
      val re = $(pattern).r
      val _content = if ($(toLowercase)) text.content.toLowerCase() else text.content
      val _mask = new Array[Int](_content.length)

      val tokens = re
        .split(_content)
        .map { token =>
          IndexedToken(
            token,
            calculateIndex("begin", _mask, _content, token, text.start),
            calculateIndex("end", _mask, _content, token, text.start))
        }
        .filter(t =>
          t.token.nonEmpty && t.token.length >= $(minLength) && get(maxLength).forall(m =>
            t.token.length <= m))

      TokenizedSentence(tokens, text.index)
    }
  }

  /** This func generates a Seq of TokenizedSentences from a Seq of Sentences.
    *
    * @param sentences
    *   to tag
    * @return
    *   Seq of TokenizedSentence objects
    */
  def tag(sentences: Seq[Sentence]): Seq[TokenizedSentence] = {
    sentences.map { text =>
      var curPos = 0

      val re = $(pattern).r
      val str = if ($(toLowercase)) text.content.toLowerCase() else text.content
      val tokens = re
        .split(str)
        .map { token =>
          curPos = str.indexOf(token, curPos)
          val indexedTokens =
            IndexedToken(token, text.start + curPos, text.start + curPos + token.length - 1)
          curPos += token.length
          indexedTokens
        }
        .filter(t =>
          t.token.nonEmpty && t.token.length >= $(minLength) && get(maxLength).forall(m =>
            t.token.length <= m))
      TokenizedSentence(tokens, text.index)
    }
  }

  /** This func applies policies for token trimming when activated.
    *
    * @param inputTokSentences
    *   input token sentences
    * @param trimWhitespace
    *   policy to trim whitespaces in tokens
    * @param preservePosition
    *   policy to preserve indexing in tokens
    * @return
    *   Seq of TokenizedSentence objects after applied policies transformations
    */
  def applyTrimPolicies(
      inputTokSentences: Seq[TokenizedSentence],
      trimWhitespace: Boolean,
      preservePosition: Boolean): Seq[TokenizedSentence] = {

    val trimRegex = "\\s+"
    val emptyStr = ""

    val leftTrimRegex = "^\\s+"
    val rightTrimRegex = "\\s+$"

    def policiesImpl(inputTokSentence: TokenizedSentence): TokenizedSentence = {
      val newIndexedTokens: Array[IndexedToken] = inputTokSentence.indexedTokens.map {
        indexedToken =>
          val inputToken = indexedToken.token
          val trimmedToken = inputToken.replaceAll(trimRegex, emptyStr)

          if (!preservePosition) {
            val leftTrimmedToken = inputToken.replaceAll(leftTrimRegex, emptyStr)
            val beginPosOffset = inputToken.length - leftTrimmedToken.length

            val rightTrimmedToken = inputToken.replaceAll(rightTrimRegex, emptyStr)
            val endNegOffset = inputToken.length - rightTrimmedToken.length

            IndexedToken(
              trimmedToken,
              indexedToken.begin + beginPosOffset,
              indexedToken.end - endNegOffset)
          } else
            IndexedToken(trimmedToken, indexedToken.begin, indexedToken.end)
      }
      TokenizedSentence(newIndexedTokens, inputTokSentence.sentenceIndex)
    }

    if (!trimWhitespace)
      inputTokSentences
    else
      inputTokSentences.map(ts => policiesImpl(ts))
  }

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val sentences = SentenceSplit.unpack(annotations)
    val tokenized = if (getPositionalMask) tagWithPositionalMask(sentences) else tag(sentences)
    val tokenizedWithPolicies =
      applyTrimPolicies(tokenized, getTrimWhitespace, getPreservePosition)
    TokenizedWithSentence.pack(tokenizedWithPolicies)
  }
}

/** This is the companion object of [[RegexTokenizer]]. Please refer to that class for the
  * documentation.
  */
object RegexTokenizer extends DefaultParamsReadable[RegexTokenizer]
