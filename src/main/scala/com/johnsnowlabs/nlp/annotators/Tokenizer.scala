package com.johnsnowlabs.nlp.annotators

import java.util.regex.Pattern

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, TOKEN}
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import com.johnsnowlabs.nlp.util.regex.{MatchStrategy, RuleFactory}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{BooleanParam, Param, StringArrayParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

import scala.collection.mutable.ArrayBuffer

class Tokenizer(override val uid: String) extends AnnotatorApproach[TokenizerModel] {

  override val outputAnnotatorType: AnnotatorType = TOKEN

  /** A Tokenizer could require only for now a SentenceDetector annotator */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array[AnnotatorType](DOCUMENT)

  def this() = this(Identifiable.randomUID("REGEX_TOKENIZER"))

  override val description: String = "Annotator that identifies points of analysis in a useful manner"

  val exceptions: StringArrayParam = new StringArrayParam(this, "exceptions", "Words that won't be affected by tokenization rules")
  val exceptionsPath: ExternalResourceParam = new ExternalResourceParam(this, "exceptionsPath", "path to file containing list of exceptions")
  val caseSensitiveExceptions: BooleanParam = new BooleanParam(this, "caseSensitiveExceptions", "Whether to care for case sensitiveness in exceptions")
  val contextChars: StringArrayParam = new StringArrayParam(this, "contextChars", "character list used to separate from token boundaries")
  val splitChars: StringArrayParam = new StringArrayParam(this, "splitChars", "character list used to separate from the inside of tokens")
  val targetPattern: Param[String] = new Param(this, "targetPattern", "pattern to grab from text as token candidates. Defaults \\S+")
  val infixPatterns: StringArrayParam = new StringArrayParam(this, "infixPatterns", "regex patterns that match tokens within a single target. groups identify different sub-tokens. multiple defaults")
  val prefixPattern: Param[String] = new Param[String](this, "prefixPattern", "regex with groups and begins with \\A to match target prefix. Overrides contextCharacters Param")
  val suffixPattern: Param[String] = new Param[String](this, "suffixPattern", "regex with groups and ends with \\z to match target suffix. Overrides contextCharacters Param")

  def setTargetPattern(value: String): this.type = set(targetPattern, value)

  def setInfixPatterns(value: Array[String]): this.type = set(infixPatterns, value)

  def addInfixPattern(value: String): this.type = set(infixPatterns, value +: $(infixPatterns))

  def setPrefixPattern(value: String): this.type = set(prefixPattern, value)

  def setSuffixPattern(value: String): this.type = set(suffixPattern, value)

  def setExceptions(value: Array[String]): this.type = set(exceptions, value)

  def addException(value: String): this.type = set(exceptions, get(exceptions).getOrElse(Array.empty[String]) :+ value)

  def getExceptions: Array[String] = $(exceptions)

  def setExceptionsPath(path: String,
               readAs: ReadAs.Format = ReadAs.LINE_BY_LINE,
               options: Map[String, String] = Map("format" -> "text")): this.type =
    set(exceptionsPath, ExternalResource(path, readAs, options))

  def setCaseSensitiveExceptions(value: Boolean): this.type = set(caseSensitiveExceptions, value)

  def getCaseSensitiveExceptions(value: Boolean): Boolean = $(caseSensitiveExceptions)

  def getInfixPatterns: Array[String] = $(infixPatterns)

  def getPrefixPattern: String = $(prefixPattern)

  def getSuffixPattern: String = $(suffixPattern)

  def getTargetPattern: String = $(targetPattern)

  def setContextChars(v: Array[String]): this.type = {
    require(v.forall(_.length == 1), "All elements in context chars must have length == 1")
    set(contextChars, v)
  }

  def addContextChars(v: String): this.type = {
    require(v.length == 1, "Context char must have length == 1")
    set(contextChars, get(contextChars).getOrElse(Array.empty[String]) :+ v)
  }

  def getContextChars: Array[String] = {
    $(contextChars)
  }

  def setSplitChars(v: Array[String]): this.type = {
    require(v.forall(_.length == 1), "All elements in context chars must have length == 1")
    set(splitChars, v)
  }

  def addSplitChars(v: String): this.type = {
    require(v.length == 1, "Context char must have length == 1")
    set(splitChars, get(splitChars).getOrElse(Array.empty[String]) :+ v)
  }

  def getSplitChars: Array[String] = {
    $(splitChars)
  }

  def buildRuleFactory: RuleFactory = {
    val rules = ArrayBuffer.empty[String]

    lazy val quotedContext = Pattern.quote($(contextChars).mkString(""))
    lazy val quotedSplit = get(splitChars).map(i => Pattern.quote(i.mkString("")))
    lazy val quotedUniqueAll = Pattern.quote(get(splitChars).getOrElse(Array.empty[String]).union($(contextChars)).distinct.mkString(""))

    val processedPrefix = get(prefixPattern).getOrElse(s"\\A([$quotedContext]*)")
    require(processedPrefix.startsWith("\\A"), "prefixPattern must begin with \\A to ensure it is the beginning of the string")

    val processedSuffix = get(suffixPattern).getOrElse(s"([$quotedContext]*)\\z")
    require(processedSuffix.endsWith("\\z"), "suffixPattern must end with \\z to ensure it is the end of the string")

    val processedInfixes = get(infixPatterns).getOrElse({
      quotedSplit
        .map(split => Array(s"([^$quotedUniqueAll]+)([$split]+)([^$quotedUniqueAll]*)"))
        .getOrElse(Array.empty[String]) ++ Array(s"([^$quotedContext](?:.*[^$quotedContext])*)")
    })

    require(processedInfixes.forall(ip => ip.contains("(") && ip.contains(")")),
      "infix patterns must use regex group. Notice each group will result in separate token")
    processedInfixes.foreach(infix => {
      val ruleBuilder = new StringBuilder
      ruleBuilder.append(processedPrefix)
      ruleBuilder.append(infix)
      ruleBuilder.append(processedSuffix)
      rules.append(ruleBuilder.toString)
    })
    rules.foldLeft(new RuleFactory(MatchStrategy.MATCH_FIRST))((factory, rule) => factory.addRule(rule.r, rule))
  }

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): TokenizerModel = {
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

    if (processedExceptions.nonEmpty)
      raw.setExceptions(processedExceptions)
    else
      raw

  }

}

object Tokenizer extends DefaultParamsReadable[Tokenizer]