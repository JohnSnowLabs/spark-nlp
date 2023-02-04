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

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, TOKEN}
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import com.johnsnowlabs.nlp.util.regex.{MatchStrategy, RuleFactory}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{BooleanParam, IntParam, Param, StringArrayParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

import java.util.regex.Pattern
import scala.collection.mutable.ArrayBuffer

/** Tokenizes raw text in document type columns into TokenizedSentence .
  *
  * This class represents a non fitted tokenizer. Fitting it will cause the internal RuleFactory
  * to construct the rules for tokenizing from the input configuration.
  *
  * Identifies tokens with tokenization open standards. A few rules will help customizing it if
  * defaults do not fit user needs.
  *
  * For extended examples of usage see the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/document-normalizer/document_normalizer_notebook.ipynb Examples]]
  * and
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/TokenizerTestSpec.scala Tokenizer test class]]
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotators.Tokenizer
  * import org.apache.spark.ml.Pipeline
  *
  * val data = Seq("I'd like to say we didn't expect that. Jane's boyfriend.").toDF("text")
  * val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("document")
  * val tokenizer = new Tokenizer().setInputCols("document").setOutputCol("token").fit(data)
  *
  * val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer)).fit(data)
  * val result = pipeline.transform(data)
  *
  * result.selectExpr("token.result").show(false)
  * +-----------------------------------------------------------------------+
  * |output                                                                 |
  * +-----------------------------------------------------------------------+
  * |[I'd, like, to, say, we, didn't, expect, that, ., Jane's, boyfriend, .]|
  * +-----------------------------------------------------------------------+
  * }}}
  *
  * @param uid
  *   required uid for storing annotator to disk
  * @groupname anno Annotator types
  * @groupdesc anno
  *   Required input and expected output annotator types
  * @groupname Ungrouped Members
  * @groupname param Parameters.
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
class Tokenizer(override val uid: String) extends AnnotatorApproach[TokenizerModel] {

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

  /** Annotator that identifies points of analysis in a useful manner */
  override val description: String =
    "Annotator that identifies points of analysis in a useful manner"

  /** Words that won't be affected by tokenization rules
    *
    * @group param
    */
  val exceptions: StringArrayParam =
    new StringArrayParam(this, "exceptions", "Words that won't be affected by tokenization rules")

  /** Path to file containing list of exceptions
    *
    * @group param
    */
  val exceptionsPath: ExternalResourceParam = new ExternalResourceParam(
    this,
    "exceptionsPath",
    "Path to file containing list of exceptions")

  /** Whether to care for case sensitiveness in exceptions (Default: `true`)
    *
    * @group param
    */
  val caseSensitiveExceptions: BooleanParam = new BooleanParam(
    this,
    "caseSensitiveExceptions",
    "Whether to care for case sensitiveness in exceptions")

  /** Character list used to separate from token boundaries (Default: `Array(".", ",", ";", ":",
    * "!", "?", "*", "-", "(", ")", "\"", "'")`)
    * @group param
    */
  val contextChars: StringArrayParam = new StringArrayParam(
    this,
    "contextChars",
    "Character list used to separate from token boundaries")

  /** Character list used to separate from the inside of tokens
    *
    * @group param
    */
  val splitChars: StringArrayParam = new StringArrayParam(
    this,
    "splitChars",
    "Character list used to separate from the inside of tokens")

  /** Pattern to separate from the inside of tokens. takes priority over splitChars.
    *
    * This pattern will be applied to the tokens which where extracted with the target pattern
    * previously
    *
    * ''' Example:'''
    *
    * {{{
    * import org.apache.spark.ml.Pipeline
    *
    * import com.johnsnowlabs.nlp.annotators.Tokenizer
    *
    * import com.johnsnowlabs.nlp.DocumentAssembler
    *
    * val textDf = sqlContext.sparkContext.parallelize(Array("Tokens in this-text will#be#split on hashtags-and#dashes")).toDF("text")
    *
    * val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("sentences")
    *
    * val tokenizer = new Tokenizer().setInputCols("sentences").setOutputCol("tokens").setSplitPattern("-|#")
    *
    * new Pipeline().setStages(Array(documentAssembler, tokenizer)).fit(textDf).transform(textDf).select("tokens.result").show(false)
    * }}}
    *
    * This will yield: `Tokens, in, this, text, will, be, split, on, hashtags, and, dashes`
    * @group param
    */
  val splitPattern: Param[String] = new Param(
    this,
    "splitPattern",
    "Pattern to separate from the inside of tokens. takes priority over splitChars.")

  /** Pattern to grab from text as token candidates. (Default: `"\\S+"`)
    *
    * Defaults to: "\\S+" which means anything not a space will be matched and considered as a
    * token candidate, This will cause text to be split on on white spaces to yield token
    * candidates.
    *
    * This rule will be added to the BREAK_PATTERN varaible, which is used to yield token
    * candidates.
    *
    * {{{
    * import org.apache.spark.ml.Pipeline
    * import com.johnsnowlabs.nlp.annotators.Tokenizer
    * import com.johnsnowlabs.nlp.DocumentAssembler
    *
    * val textDf = sqlContext.sparkContext.parallelize(Array("I only consider lowercase characters and NOT UPPERCASED and only the numbers 0,1, to 7 as tokens but not 8 or 9")).toDF("text")
    * val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("sentences")
    * val tokenizer = new Tokenizer().setInputCols("sentences").setOutputCol("tokens").setTargetPattern("a-z-0-7")
    * new Pipeline().setStages(Array(documentAssembler, tokenizer)).fit(textDf).transform(textDf).select("tokens.result").show(false)
    * }}}
    *
    * This will yield: `only, consider, lowercase, characters, and, and, only, the, numbers, 0, 1,
    * to, 7, as, tokens, but, not, or`
    * @group param
    */
  val targetPattern: Param[String] = new Param(
    this,
    "targetPattern",
    "Pattern to grab from text as token candidates. Defaults \\S+")

  /** Regex patterns that match tokens within a single target. groups identify different
    * sub-tokens. multiple defaults
    *
    * Infix patterns must use regex group. Notice each group will result in separate token
    *
    * '''Example:'''
    *
    * {{{
    * import org.apache.spark.ml.Pipeline
    * import com.johnsnowlabs.nlp.annotators.Tokenizer
    * import com.johnsnowlabs.nlp.DocumentAssembler
    *
    * val textDf = sqlContext.sparkContext.parallelize(Array("l'une d'un l'un, des l'extrême des l'extreme")).toDF("text")
    * val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("sentences")
    * val tokenizer = new Tokenizer().setInputCols("sentences").setOutputCol("tokens").setInfixPatterns(Array("([\\p{L}\\w]+'{1})([\\p{L}\\w]+)"))
    * new Pipeline().setStages(Array(documentAssembler, tokenizer)).fit(textDf).transform(textDf).select("tokens.result").show(false)
    *
    * }}}
    *
    * This will yield: `l', une, d', un, l', un, , , des, l', extrême, des, l', extreme`
    * @group param
    */
  val infixPatterns: StringArrayParam = new StringArrayParam(
    this,
    "infixPatterns",
    "Regex patterns that match tokens within a single target. groups identify different sub-tokens. multiple defaults")

  /** Regex with groups and begins with \\A to match target prefix. Overrides contextCharacters
    * Param
    *
    * @group param
    */
  val prefixPattern: Param[String] = new Param[String](
    this,
    "prefixPattern",
    "Regex with groups and begins with \\A to match target prefix. Overrides contextCharacters Param")

  /** Regex with groups and ends with \\z to match target suffix. Overrides contextCharacters
    * Param
    *
    * @group param
    */
  val suffixPattern: Param[String] = new Param[String](
    this,
    "suffixPattern",
    "Regex with groups and ends with \\z to match target suffix. Overrides contextCharacters Param")

  /** Set the minimum allowed length for each token
    *
    * @group param
    */
  val minLength = new IntParam(this, "minLength", "Set the minimum allowed length for each token")

  /** Set the minimum allowed length for each token
    * @group setParam
    */
  def setMinLength(value: Int): this.type = {
    require(value >= 0, "minLength must be greater equal than 0")
    require(value.isValidInt, "minLength must be Int")
    set(minLength, value)
  }

  /** Get the minimum allowed length for each token
    * @group getParam
    */
  def getMinLength(value: Int): Int = $(minLength)

  /** Set the maximum allowed length for each token
    *
    * @group param
    */
  val maxLength = new IntParam(this, "maxLength", "Set the maximum allowed length for each token")

  /** Get the maximum allowed length for each token
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

  /** Get the maximum allowed length for each token
    * @group getParam
    */
  def getMaxLength(value: Int): Int = $(maxLength)

  /** Set a basic regex rule to identify token candidates in text.
    * @group setParam
    */
  def setTargetPattern(value: String): this.type = set(targetPattern, value)

  /** Regex pattern to separate from the inside of tokens. Takes priority over splitChars.
    * @group setParam
    */
  def setSplitPattern(value: String): this.type = set(splitPattern, value)

  /** Set a list of Regex patterns that match tokens within a single target. Groups identify
    * different sub-tokens. multiple defaults
    * @group setParam
    */
  def setInfixPatterns(value: Array[String]): this.type = set(infixPatterns, value)

  /** Add an extension pattern regex with groups to the top of thsetExceptionse rules (will target
    * first, from more specific to the more general).
    *
    * @group setParam
    */
  def addInfixPattern(value: String): this.type = set(infixPatterns, value +: $(infixPatterns))

  /** Regex to identify subtokens that come in the beginning of the token. Regex has to start with
    * \\A and must contain groups (). Each group will become a separate token within the prefix.
    * Defaults to non-letter characters. e.g. quotes or parenthesis
    *
    * @group setParam
    */
  def setPrefixPattern(value: String): this.type = set(prefixPattern, value)

  /** Regex to identify subtokens that are in the end of the token. Regex has to end with \\z and
    * must contain groups (). Each group will become a separate token within the prefix. Defaults
    * to non-letter characters. e.g. quotes or parenthesis
    *
    * @group setParam
    */
  def setSuffixPattern(value: String): this.type = set(suffixPattern, value)

  /** List of tokens to not alter at all. Allows composite tokens like two worded tokens that the
    * user may not want to split.
    *
    * @group setParam
    */
  def setExceptions(value: Array[String]): this.type = set(exceptions, value)

  /** Add a single exception
    *
    * @group setParam
    */
  def addException(value: String): this.type =
    set(exceptions, get(exceptions).getOrElse(Array.empty[String]) :+ value)

  /** @group setParam */
  def getExceptions: Array[String] = $(exceptions)

  /** Path to txt file with list of token exceptions
    *
    * @group getParam
    */
  def setExceptionsPath(
      path: String,
      readAs: ReadAs.Format = ReadAs.TEXT,
      options: Map[String, String] = Map("format" -> "text")): this.type =
    set(exceptionsPath, ExternalResource(path, readAs, options))

  /** Whether to follow case sensitiveness for matching exceptions in text
    *
    * @group getParam
    */
  def setCaseSensitiveExceptions(value: Boolean): this.type = set(caseSensitiveExceptions, value)

  /** Whether to follow case sensitiveness for matching exceptions in text
    *
    * @group getParam
    */
  def getCaseSensitiveExceptions(value: Boolean): Boolean = $(caseSensitiveExceptions)

  /** Add an extension pattern regex with groups to the top of the rules (will target first, from
    * more specific to the more general).
    *
    * @group getParam
    */
  def getInfixPatterns: Array[String] = $(infixPatterns)

  /** Regex to identify subtokens that come in the beginning of the token. Regex has to start with
    * \\A and must contain groups (). Each group will become a separate token within the prefix.
    * Defaults to non-letter characters. e.g. quotes or parenthesis
    *
    * @group getParam
    */
  def getPrefixPattern: String = $(prefixPattern)

  /** Regex to identify subtokens that are in the end of the token. Regex has to end with \\z and
    * must contain groups (). Each group will become a separate token within the prefix. Defaults
    * to non-letter characters. e.g. quotes or parenthesis
    *
    * @group getParam
    */
  def getSuffixPattern: String = $(suffixPattern)

  /** Basic regex rule to identify a candidate for tokenization. Defaults to \\S+ which means
    * anything not a space
    *
    * @group getParam
    */
  def getTargetPattern: String = $(targetPattern)

  /** List of 1 character string to split tokens inside, such as hyphens. Ignored if using infix,
    * prefix or suffix patterns.
    *
    * @group getParam
    */
  def getSplitPattern: String = $(splitPattern)

  /** List of 1 character string to rip off from tokens, such as parenthesis or question marks.
    * Ignored if using prefix, infix or suffix patterns.
    *
    * @group setParam
    */
  def setContextChars(v: Array[String]): this.type = {
    require(v.forall(_.length == 1), "All elements in context chars must have length == 1")
    set(contextChars, v)
  }

  /** Add one character string to rip off from tokens, such as parenthesis or question marks.
    * Ignored if using prefix, infix or suffix patterns.
    *
    * @group setParam
    */
  def addContextChars(v: String): this.type = {
    require(v.length == 1, "Context char must have length == 1")
    set(contextChars, get(contextChars).getOrElse(Array.empty[String]) :+ v)
  }

  /** List of 1 character string to rip off from tokens, such as parenthesis or question marks.
    * Ignored if using prefix, infix or suffix patterns.
    *
    * @group getParam
    */
  def getContextChars: Array[String] = {
    $(contextChars)
  }

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
    * prefix or suffix patterns.
    *
    * @group getParam
    */
  def getSplitChars: Array[String] = {
    $(splitChars)
  }

  setDefault(
    inputCols -> Array(DOCUMENT),
    outputCol -> "token",
    targetPattern -> "\\S+",
    contextChars -> Array(".", ",", ";", ":", "!", "?", "*", "-", "(", ")", "\"", "'"),
    caseSensitiveExceptions -> true,
    minLength -> 0)

  /** Build rule factory which combines all defined parameters to build regex that is applied to
    * tokens
    */
  def buildRuleFactory: RuleFactory = {
    val rules = ArrayBuffer.empty[String]

    lazy val quotedContext = Pattern.quote($(contextChars).mkString(""))

    val processedPrefix = get(prefixPattern).getOrElse(s"\\A([$quotedContext]*)")
    require(
      processedPrefix.startsWith("\\A"),
      "prefixPattern must begin with \\A to ensure it is the beginning of the string")

    val processedSuffix = get(suffixPattern).getOrElse(s"([$quotedContext]*)\\z")
    require(
      processedSuffix.endsWith("\\z"),
      "suffixPattern must end with \\z to ensure it is the end of the string")

    val processedInfixes =
      get(infixPatterns).getOrElse(Array(s"([^$quotedContext](?:.*[^$quotedContext])*)"))

    require(
      processedInfixes.forall(ip => ip.contains("(") && ip.contains(")")),
      "infix patterns must use regex group. Notice each group will result in separate token")
    processedInfixes.foreach(infix => {
      val ruleBuilder = new StringBuilder
      ruleBuilder.append(processedPrefix)
      ruleBuilder.append(infix)
      ruleBuilder.append(processedSuffix)
      rules.append(ruleBuilder.toString)
    })
    rules.foldLeft(new RuleFactory(MatchStrategy.MATCH_FIRST))((factory, rule) =>
      factory.addRule(rule.r, rule))
  }

  /** Clears out rules and constructs a new rule for every combination of rules provided . The
    * strategy is to catch one token per regex group. User may add its own groups if needs targets
    * to be tokenized separately from the rest
    */
  override def train(
      dataset: Dataset[_],
      recursivePipeline: Option[PipelineModel]): TokenizerModel = {

    /** Clears out rules and constructs a new rule for every combination of rules provided */
    /** The strategy is to catch one token per regex group */
    /** User may add its own groups if needs targets to be tokenized separately from the rest */
    val ruleFactory = buildRuleFactory

    val processedExceptions = get(exceptionsPath)
      .map(er => ResourceHelper.parseLines(er))
      .getOrElse(Array.empty[String]) ++ get(exceptions).getOrElse(Array.empty[String])

    val raw = new TokenizerModel()
      .setCaseSensitiveExceptions($(caseSensitiveExceptions))
      .setTargetPattern($(targetPattern))
      .setRules(ruleFactory)
      .setMinLength($(minLength))

    if (isDefined(maxLength))
      raw.setMaxLength($(maxLength))

    if (processedExceptions.nonEmpty)
      raw.setExceptions(processedExceptions)

    if (isSet(splitPattern)) raw.setSplitPattern($(splitPattern))

    if (isSet(splitChars)) raw.setSplitChars($(splitChars))

    raw

  }

}

/** This is the companion object of [[Tokenizer]]. Please refer to that class for the
  * documentation.
  */
object Tokenizer extends DefaultParamsReadable[Tokenizer]
