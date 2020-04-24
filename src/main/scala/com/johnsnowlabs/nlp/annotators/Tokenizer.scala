package com.johnsnowlabs.nlp.annotators

import java.util.regex.Pattern

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, TOKEN}
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import com.johnsnowlabs.nlp.util.regex.{MatchStrategy, RuleFactory}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{BooleanParam, IntParam, Param, StringArrayParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

import scala.collection.mutable.ArrayBuffer

/**
  * Tokenizes raw text in document type columns into TokenizedSentence .
  * This class represents a non fitted tokenizer. Fitting it will cause the internal RuleFactory to construct the rules for tokenizing from the input configuration.
  *
  * See [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/TokenizerTestSpec.scala Tokenizer test class]] for examples examples of usage.
  *
  * @param uid
  */
class Tokenizer(override val uid: String) extends AnnotatorApproach[TokenizerModel] {

  override val outputAnnotatorType: AnnotatorType = TOKEN

  /** A Tokenizer could require only for now a SentenceDetector annotator */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array[AnnotatorType](DOCUMENT)

  def this() = this(Identifiable.randomUID("REGEX_TOKENIZER"))

  override val description: String = "Annotator that identifies points of analysis in a useful manner"

  val exceptions: StringArrayParam = new StringArrayParam(this, "exceptions", "Words that won't be affected by tokenization rules")
  val exceptionsPath: ExternalResourceParam = new ExternalResourceParam(this, "exceptionsPath", "Path to file containing list of exceptions")
  val caseSensitiveExceptions: BooleanParam = new BooleanParam(this, "caseSensitiveExceptions", "Whether to care for case sensitiveness in exceptions")
  val contextChars: StringArrayParam = new StringArrayParam(this, "contextChars", "Character list used to separate from token boundaries")
  val splitChars: StringArrayParam = new StringArrayParam(this, "splitChars", "Character list used to separate from the inside of tokens")
  val splitPattern: Param[String] = new Param(this, "splitPattern", "Pattern to separate from the inside of tokens. takes priority over splitChars.")
  val targetPattern: Param[String] = new Param(this, "targetPattern", "Pattern to grab from text as token candidates. Defaults \\S+")
  val infixPatterns: StringArrayParam = new StringArrayParam(this, "infixPatterns", "Regex patterns that match tokens within a single target. groups identify different sub-tokens. multiple defaults")
  val prefixPattern: Param[String] = new Param[String](this, "prefixPattern", "Regex with groups and begins with \\A to match target prefix. Overrides contextCharacters Param")
  val suffixPattern: Param[String] = new Param[String](this, "suffixPattern", "Regex with groups and ends with \\z to match target suffix. Overrides contextCharacters Param")
  val minLength = new IntParam(this, "minLength", "Set the minimum allowed length for each token")
  val maxLength = new IntParam(this, "maxLength", "Set the maximum allowed length for each token")


  /**
    * Set a basic regex rule to identify token candidates in text.
    *
    * Defaults to: "\\S+" which means anything not a space will be matched and considered as a token candidate, This will cause text to be split on on white spaces  to yield token candidates.
    *
    * This rule will be added to the BREAK_PATTERN varaible, which is used to yield token candidates.
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
    * This will yield : [only, consider, lowercase, characters, and, and, only, the, numbers, 0, 1, to, 7, as, tokens, but, not, or]
    */
  def setTargetPattern(value: String): this.type = set(targetPattern, value)

  /**
    * Regex pattern to separate from the inside of tokens. Takes priority over splitChars.
    *
    * This pattern will be applied to the tokens which where extracted with the target pattern previously
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
    * This will yield : [Tokens, in, this, text, will, be, split, on, hashtags, and, dashes]
    *
    */
  def setSplitPattern(value: String): this.type = set(splitPattern, value)


  /**
    * Set a list of Regex patterns that match tokens within a single target. Groups identify different sub-tokens. multiple defaults
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
    * This will yield [l', une, d', un, l', un, , , des, l', extrême, des, l', extreme]
    *
    */
  def setInfixPatterns(value: Array[String]): this.type = set(infixPatterns, value)

  /**
    * Add an extension pattern regex with groups to the top of the rules (will target first, from more specific to the more general).
    */
  def addInfixPattern(value: String): this.type = set(infixPatterns, value +: $(infixPatterns))


  def setPrefixPattern(value: String): this.type = set(prefixPattern, value)

  def setSuffixPattern(value: String): this.type = set(suffixPattern, value)

  def setExceptions(value: Array[String]): this.type = set(exceptions, value)

  def addException(value: String): this.type = set(exceptions, get(exceptions).getOrElse(Array.empty[String]) :+ value)

  def getExceptions: Array[String] = $(exceptions)

  def setExceptionsPath(path: String,
               readAs: ReadAs.Format = ReadAs.TEXT,
               options: Map[String, String] = Map("format" -> "text")): this.type =
    set(exceptionsPath, ExternalResource(path, readAs, options))

  def setCaseSensitiveExceptions(value: Boolean): this.type = set(caseSensitiveExceptions, value)

  def getCaseSensitiveExceptions(value: Boolean): Boolean = $(caseSensitiveExceptions)

  def getInfixPatterns: Array[String] = $(infixPatterns)

  def getPrefixPattern: String = $(prefixPattern)

  def getSuffixPattern: String = $(suffixPattern)

  def getTargetPattern: String = $(targetPattern)

  def getSplitPattern: String = $(splitPattern)

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
    require(v.forall(x=>x.length == 1 || (x.length==2 && x.substring(0,1)=="\\")), "All elements in context chars must have length == 1")
    set(splitChars, v)
  }

  def addSplitChars(v: String): this.type = {
    require(v.length == 1 || (v.length==2 && v.substring(0,1)=="\\"), "Context char must have length == 1")
    set(splitChars, get(splitChars).getOrElse(Array.empty[String]) :+ v)
  }

  def getSplitChars: Array[String] = {
    $(splitChars)
  }

  def setMinLength(value: Int): this.type = {
    require(value >= 0, "minLength must be greater equal than 0")
    require(value.isValidInt, "minLength must be Int")
    set(minLength, value)
  }
  def getMinLength(value: Int): Int = $(minLength)

  def setMaxLength(value: Int): this.type = {
    require(value >= ${minLength}, "maxLength must be greater equal than minLength")
    require(value.isValidInt, "minLength must be Int")
    set(maxLength, value)
  }
  def getMaxLength(value: Int): Int = $(maxLength)

  setDefault(
    targetPattern -> "\\S+",
    contextChars -> Array(".", ",", ";", ":", "!", "?", "*", "-", "(", ")", "\"", "'"),
    caseSensitiveExceptions -> true,
    minLength -> 0
  )

  def buildRuleFactory: RuleFactory = {
    val rules = ArrayBuffer.empty[String]

    lazy val quotedContext = Pattern.quote($(contextChars).mkString(""))

    val processedPrefix = get(prefixPattern).getOrElse(s"\\A([$quotedContext]*)")
    require(processedPrefix.startsWith("\\A"), "prefixPattern must begin with \\A to ensure it is the beginning of the string")

    val processedSuffix = get(suffixPattern).getOrElse(s"([$quotedContext]*)\\z")
    require(processedSuffix.endsWith("\\z"), "suffixPattern must end with \\z to ensure it is the end of the string")

    val processedInfixes = get(infixPatterns).getOrElse(Array(s"([^$quotedContext](?:.*[^$quotedContext])*)"))

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

object Tokenizer extends DefaultParamsReadable[Tokenizer]