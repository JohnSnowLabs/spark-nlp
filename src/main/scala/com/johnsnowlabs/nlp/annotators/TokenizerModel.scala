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

import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.serialization.StructFeature
import com.johnsnowlabs.nlp.util.regex.RuleFactory
import org.apache.spark.ml.param.{BooleanParam, IntParam, Param, StringArrayParam}
import org.apache.spark.ml.util.Identifiable

import scala.collection.mutable
import scala.util.matching.Regex

/** Tokenizes raw text into word pieces, tokens. Identifies tokens with tokenization open
  * standards. A few rules will help customizing it if defaults do not fit user needs.
  *
  * This class represents an already fitted Tokenizer model.
  *
  * See the main class [[Tokenizer]] for more examples of usage.
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
class TokenizerModel(override val uid: String)
    extends AnnotatorModel[TokenizerModel]
    with HasSimpleAnnotate[TokenizerModel] {

  import com.johnsnowlabs.nlp.AnnotatorType._

  /** rules
    *
    * @group param
    */
  val rules: StructFeature[RuleFactory] = new StructFeature[RuleFactory](this, "rules")

  /** Words that won't be affected by tokenization rules
    *
    * @group param
    */
  val exceptions: StringArrayParam =
    new StringArrayParam(this, "exceptions", "Words that won't be affected by tokenization rules")

  /** Whether to care for case sensitiveness in exceptions (Default: `true`)
    *
    * @group param
    */
  val caseSensitiveExceptions: BooleanParam = new BooleanParam(
    this,
    "caseSensitiveExceptions",
    "Whether to care for case sensitiveness in exceptions")

  /** pattern to grab from text as token candidates. Defaults \\S+
    *
    * @group param
    */
  val targetPattern: Param[String] = new Param(
    this,
    "targetPattern",
    "pattern to grab from text as token candidates. Defaults \\S+")

  /** Set the minimum allowed length for each token
    *
    * @group param
    */
  val minLength = new IntParam(this, "minLength", "Set the minimum allowed length for each token")

  /** Set the maximum allowed length for each token
    *
    * @group param
    */
  val maxLength = new IntParam(this, "maxLength", "Set the maximum allowed length for each token")

  /** character list used to separate from the inside of tokens
    *
    * @group param
    */
  val splitChars: StringArrayParam = new StringArrayParam(
    this,
    "splitChars",
    "character list used to separate from the inside of tokens")

  /** pattern to separate from the inside of tokens. takes priority over splitChars.
    *
    * @group param
    */
  val splitPattern: Param[String] = new Param(
    this,
    "splitPattern",
    "pattern to separate from the inside of tokens. takes priority over splitChars.")

  setDefault(targetPattern -> "\\S+", caseSensitiveExceptions -> true)

  /** Output annotator type : TOKEN
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = TOKEN

  /** Input annotator type : DOCUMENT
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array[AnnotatorType](
    DOCUMENT
  ) // A Tokenizer could require only for now a SentenceDetector annotator

  def this() = this(Identifiable.randomUID("REGEX_TOKENIZER"))

  /** pattern to grab from text as token candidates. Defaults \\S+
    *
    * @group setParam
    */
  def setTargetPattern(value: String): this.type = set(targetPattern, value)

  /** pattern to grab from text as token candidates. Defaults \\S+
    *
    * @group getParam
    */
  def getTargetPattern: String = $(targetPattern)

  /** List of 1 character string to split tokens inside, such as hyphens. Ignored if using infix,
    * prefix or suffix patterns.
    *
    * @group setParam
    */
  def setSplitPattern(value: String): this.type = set(splitPattern, value)

  /** List of 1 character string to split tokens inside, such as hyphens. Ignored if using infix,
    * prefix or suffix patterns.
    *
    * @group getParam
    */
  def getSplitPattern: String = $(splitPattern)

  /** Words that won't be affected by tokenization rules
    *
    * @group setParam
    */
  def setExceptions(value: Array[String]): this.type = set(exceptions, value)

  /** Words that won't be affected by tokenization rules
    *
    * @group getParam
    */
  def getExceptions: Array[String] = $(exceptions)

  /** Rules factory for tokenization
    *
    * @group setParam
    */
  def setRules(ruleFactory: RuleFactory): this.type = set(rules, ruleFactory)

  /** Whether to follow case sensitiveness for matching exceptions in text
    *
    * @group setParam
    */
  def setCaseSensitiveExceptions(value: Boolean): this.type = set(caseSensitiveExceptions, value)

  /** Whether to follow case sensitiveness for matching exceptions in text
    *
    * @group getParam
    */
  def getCaseSensitiveExceptions(value: Boolean): Boolean = $(caseSensitiveExceptions)

  /** Set the minimum allowed length for each token
    *
    * @group setParam
    */
  def setMinLength(value: Int): this.type = set(minLength, value)

  /** Set the minimum allowed length for each token
    *
    * @group getParam
    */
  def getMinLength(value: Int): Int = $(minLength)

  /** Set the maximum allowed length for each token
    *
    * @group setParam
    */
  def setMaxLength(value: Int): this.type = set(maxLength, value)

  /** Set the maximum allowed length for each token
    *
    * @group getParam
    */
  def getMaxLength(value: Int): Int = $(maxLength)

  /** List of 1 character string to split tokens inside, such as hyphens. Ignored if using infix,
    * prefix or suffix patterns.
    *
    * @group setParam
    */
  def setSplitChars(v: Array[String]): this.type = {
    require(
      v.forall(x => x.length == 1 || (x.length == 2 && x.substring(0, 1) == "\\")),
      "All elements in context chars must have length == 1")
    set(splitChars, v)
  }

  /** One character string to split tokens inside, such as hyphens. Ignored if using infix, prefix
    * or suffix patterns.
    *
    * @group setParam
    */
  def addSplitChars(v: String): this.type = {
    require(
      v.length == 1 || (v.length == 2 && v.substring(0, 1) == "\\"),
      "Context char must have length == 1")
    set(splitChars, get(splitChars).getOrElse(Array.empty[String]) :+ v)
  }

  /** List of 1 character string to split tokens inside, such as hyphens. Ignored if using infix,
    * prefix or suffix patterns
    *
    * @group getParam
    * .
    */
  def getSplitChars: Array[String] = {
    $(splitChars)
  }

  private val PROTECT_CHAR = "ↈ"
  private val BREAK_CHAR = "ↇ"

  private lazy val BREAK_PATTERN = ("[^(?:" + $(targetPattern) + ")" + PROTECT_CHAR + "]").r
  private lazy val SPLIT_PATTERN = ("[^" + BREAK_CHAR + "]+").r

  private var compiledExceptions: Option[Regex] = None

  private def getOrCompileExceptionPattern(): Regex = compiledExceptions.getOrElse {
    get(exceptions) match {
      case None => "".r
      case Some(patterns) =>
        val compiledPattern: Regex = patterns
          .map { pattern =>
            val casedExceptionPattern =
              if ($(caseSensitiveExceptions)) pattern else "(?i)" + pattern
            casedExceptionPattern
          }
          .mkString("|")
          .r
        compiledExceptions = Some(compiledPattern)
        compiledPattern
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
    lazy val splitCharsExists = $(splitChars).map(_.last.toString)
    sentences.map { text =>
      /** Step 1, define breaks from non breaks */
      val exceptionsDefined = get(exceptions).isDefined

      var exceptionsWithoutBreak: Option[mutable.HashSet[String]] = None

      val textContent = if (exceptionsDefined) {

        /** If found, replace BREAK_PATTERN with PROTECT_CHAR, otherwise add to an exception list.
          */
        getOrCompileExceptionPattern().replaceAllIn(
          text.content,
          { m: Regex.Match =>
            {
              val breakReplaced = BREAK_PATTERN.replaceAllIn(m.matched, PROTECT_CHAR)
              if (breakReplaced == m.matched)
                exceptionsWithoutBreak.getOrElse({
                  exceptionsWithoutBreak = Some(new mutable.HashSet[String])
                  exceptionsWithoutBreak.get
                }) += m.matched
              breakReplaced
            }
          })
      } else {
        text.content
      }

      val protectedText = BREAK_PATTERN.replaceAllIn(textContent, BREAK_CHAR)

      /** Step 2, Return protected tokens back into text and move on */
      val tokens = SPLIT_PATTERN
        .findAllMatchIn(protectedText)
        .flatMap { candidate =>
          /** If exceptions are defined, check for candidate whether PROTECT_CHAR present or in
            * exception list.
            */
          if (exceptionsDefined &&
            (candidate.matched.contains(PROTECT_CHAR) ||
              (exceptionsWithoutBreak.isDefined && exceptionsWithoutBreak.get.contains(
                candidate.matched)))) {

            /** Put back character and move on */
            Seq(
              IndexedToken(
                text.content.slice(candidate.start, candidate.end),
                text.start + candidate.start,
                text.start + candidate.end - 1))
          } else {

            /** Step 3, If no exception found, find candidates through the possible general rule
              * patterns
              */
            val rr = $$(rules)
              .findMatchFirstOnly(candidate.matched)
              .map { m =>
                var curPos = m.content.start
                (1 to m.content.groupCount)
                  .flatMap(i => {
                    val target = m.content.group(i)
                    val applyPattern = isSet(splitPattern) && (target
                      .split($(splitPattern))
                      .length > 1)
                    val applyChars = isSet(splitChars) && splitCharsExists.exists(target.contains)
                    if (target.nonEmpty && (applyPattern || applyChars)) {
                      try {
                        val strs =
                          if (applyPattern) target.split($(splitPattern))
                          else target.split($(splitChars).mkString("|"))
                        strs.map(str =>
                          try {
                            IndexedToken(
                              str,
                              text.start + candidate.start + curPos,
                              text.start + candidate.start + curPos + str.length - 1)
                          } finally {
                            curPos += str.length + 1
                          })
                      } finally {
                        curPos -= 1
                      }
                    } else {
                      val it = IndexedToken(
                        target,
                        text.start + candidate.start + curPos,
                        text.start + candidate.start + curPos + target.length - 1)
                      curPos += target.length
                      Seq(it)
                    }
                  })

              /** Step 4, If rules didn't match, return whatever candidate we have and leave it as
                * is
                */
              }
              .getOrElse(Seq(IndexedToken(
                candidate.matched,
                text.start + candidate.start,
                text.start + candidate.end - 1)))
            rr
          }
        }
        .filter(t =>
          t.token.nonEmpty && t.token.length >= $(minLength) && get(maxLength).forall(m =>
            t.token.length <= m))
        .toArray
      TokenizedSentence(tokens, text.index)
    }
  }

  /** one to many annotation */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val sentences = SentenceSplit.unpack(annotations)
    val tokenized = tag(sentences)
    TokenizedWithSentence.pack(tokenized)
  }
}

trait ReadablePretrainedTokenizer
    extends ParamsAndFeaturesReadable[TokenizerModel]
    with HasPretrained[TokenizerModel] {
  override val defaultModelName: Option[String] = Some("token_rules")

  /** Java compliant-overrides */
  override def pretrained(): TokenizerModel = super.pretrained()

  override def pretrained(name: String): TokenizerModel = super.pretrained(name)

  override def pretrained(name: String, lang: String): TokenizerModel =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): TokenizerModel =
    super.pretrained(name, lang, remoteLoc)
}

/** This is the companion object of [[TokenizerModel]]. Please refer to that class for the
  * documentation.
  */
object TokenizerModel extends ReadablePretrainedTokenizer
