package com.johnsnowlabs.nlp.annotators

import java.util.regex.Pattern

import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.util.regex.{MatchStrategy, RuleFactory}
import org.apache.spark.ml.param.{Param, StringArrayParam}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

import scala.collection.mutable.ArrayBuffer

/**
  * Tokenizes raw text into word pieces, tokens.
  * @param uid required uid for storing annotator to disk
  * @@ pattern: RegexPattern to split phrases into tokens
  */
class Tokenizer(override val uid: String) extends AnnotatorModel[Tokenizer] {

  import com.johnsnowlabs.nlp.AnnotatorType._

  val compositeTokens: StringArrayParam = new StringArrayParam(this, "compositeTokens", "Words that won't be split in two")
  val exceptionTokens: StringArrayParam = new StringArrayParam(this, "exceptionTokens", "Words that won't be affected by tokenization rules")
  val contextChars: StringArrayParam = new StringArrayParam(this, "contextChars", "character list used to separate from token boundaries")
  val splitChars: StringArrayParam = new StringArrayParam(this, "splitChars", "character list used to separate from the inside of tokens")
  val targetPattern: Param[String] = new Param(this, "targetPattern", "pattern to grab from text as token candidates. Defaults \\S+")
  val infixPatterns: StringArrayParam = new StringArrayParam(this, "infixPatterns", "regex patterns that match tokens within a single target. groups identify different sub-tokens. multiple defaults")
  val prefixPattern: Param[String] = new Param[String](this, "prefixPattern", "regex with groups and begins with \\A to match target prefix. Overrides contextCharacters Param")
  val suffixPattern: Param[String] = new Param[String](this, "suffixPattern", "regex with groups and ends with \\z to match target suffix. Overrides contextCharacters Param")

  override val outputAnnotatorType: AnnotatorType = TOKEN

  /** A Tokenizer could require only for now a SentenceDetector annotator */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array[AnnotatorType](DOCUMENT)

  def this() = this(Identifiable.randomUID("REGEX_TOKENIZER"))

  def setTargetPattern(value: String): this.type = set(targetPattern, value)

  def setInfixPatterns(value: Array[String]): this.type = set(infixPatterns, value)

  def addInfixPattern(value: String): this.type = set(infixPatterns, value +: $(infixPatterns))

  def setPrefixPattern(value: String): this.type = set(prefixPattern, value)

  def setSuffixPattern(value: String): this.type = set(suffixPattern, value)

  def setCompositeTokens(value: Array[String]): this.type = set(compositeTokens, value)

  def addCompositeTokens(value: String): this.type = set(compositeTokens, get(compositeTokens).getOrElse(Array.empty[String] :+ value))

  def getCompositeTokens: Array[String] = $(compositeTokens)

  def setExceptionTokens(value: Array[String]): this.type = set(compositeTokens, value)

  def addExceptionTokens(value: String): this.type = set(exceptionTokens, get(exceptionTokens).getOrElse(Array.empty[String]) :+ value)

  def getExceptionTokens: Array[String] = $(compositeTokens)

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

  setDefault(
    targetPattern -> "\\S+",
    contextChars -> Array(".", ",", ";", ":", "!", "?", "*", "-", "(", ")", "\"", "'")
  )

  /** Clears out rules and constructs a new rule for every combination of rules provided */
  /** The strategy is to catch one token per regex group */
  /** User may add its own groups if needs targets to be tokenized separately from the rest */
  lazy private val ruleFactory = {
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

  private val PROTECT_CHAR = "ↈ"
  private val BREAK_CHAR = "ↇ"

  private lazy val BREAK_PATTERN = "[^(?:" + $(targetPattern) + ")" + PROTECT_CHAR + "]"
  private lazy val SPLIT_PATTERN = "[^" + BREAK_CHAR + "]+"

  def tag(sentences: Seq[Sentence]): Seq[TokenizedSentence] = {
    sentences.map{text =>
      /** Step 1, define breaks from non breaks */
      val protectedText = {
        get(compositeTokens).map(_.foldRight(text.content)((compositeToken, currentText) => {
          currentText.replaceAll(
            compositeToken,
            compositeToken.replaceAll(BREAK_PATTERN, PROTECT_CHAR)
          )
        })).getOrElse(text.content).replaceAll(BREAK_PATTERN, BREAK_CHAR)
      }
      /** Step 2, Return protected tokens back into text and move on*/
      val tokens = SPLIT_PATTERN.r.findAllMatchIn(protectedText).flatMap { candidate =>
        if (get(compositeTokens).isDefined && candidate.matched.contains(PROTECT_CHAR)) {
          /** Put back character and move on */
          Seq(IndexedToken(
            text.content.slice(text.start + candidate.start, text.start + candidate.end),
            text.start + candidate.start,
            text.start + candidate.end - 1
          ))
        } else if (get(exceptionTokens).isDefined && $(exceptionTokens).contains(candidate.matched)) {
          Seq(IndexedToken(
            candidate.matched,
            candidate.start,
            candidate.end - 1
          ))
        }
        else {
        /** Step 3, If no exception found, find candidates through the possible general rule patterns*/
        ruleFactory.findMatchFirstOnly(candidate.matched).map {m =>
          var curPos = m.content.start
          (1 to m.content.groupCount)
            .map (i => {
              val target = m.content.group(i)
              val it = IndexedToken(
                target,
                text.start + candidate.start + curPos,
                text.start + candidate.start + curPos + target.length - 1
              )
              curPos += target.length
              it
            })
          /** Step 4, If rules didn't match, return whatever candidate we have and leave it as is*/
          }.getOrElse(Seq(IndexedToken(
            candidate.matched,
            text.start + candidate.start,
            text.start + candidate.end - 1
        )))
      }}.toArray.filter(t => t.token.nonEmpty)
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

object Tokenizer extends DefaultParamsReadable[Tokenizer]