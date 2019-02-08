package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.util.regex.{MatchStrategy, RuleFactory}
import org.apache.spark.ml.param.{BooleanParam, Param, StringArrayParam}
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
  val targetPattern: Param[String] = new Param(this, "targetPattern", "pattern to grab from text as token candidates. Defaults \\S+")
  val infixPatterns: StringArrayParam = new StringArrayParam(this, "infixPatterns", "regex patterns that match tokens within a single target. groups identify different sub-tokens. multiple defaults")
  val prefixPattern: Param[String] = new Param[String](this, "prefixPattern", "regex with groups and begins with \\A to match target prefix. Defaults to \\A([^\\s\\p{L}$\\.]*)")
  val suffixPattern: Param[String] = new Param[String](this, "suffixPattern", "regex with groups and ends with \\z to match target suffix. Defaults to ([^\\s\\p{L}]?)([^\\s\\p{L}]*)\\z")
  val includeDefaults: BooleanParam = new BooleanParam(this, "includeDefaults", "whether to include default patterns or only use user provided ones. Defaults to true.")

  override val annotatorType: AnnotatorType = TOKEN

  /** A Tokenizer could require only for now a SentenceDetector annotator */
  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array[AnnotatorType](DOCUMENT)

  def this() = this(Identifiable.randomUID("REGEX_TOKENIZER"))

  def setTargetPattern(value: String): this.type = set(targetPattern, value)

  def setInfixPatterns(value: Array[String]): this.type = set(infixPatterns, value)

  def addInfixPattern(value: String): this.type = set(infixPatterns, value +: $(infixPatterns))

  def setPrefixPattern(value: String): this.type = set(prefixPattern, value)

  def setSuffixPattern(value: String): this.type = set(suffixPattern, value)

  def setCompositeTokensPatterns(value: Array[String]): this.type = set(compositeTokens, value)

  def getCompositeTokens: Array[String] = $(compositeTokens)

  def getInfixPatterns: Array[String] = if ($(includeDefaults)) $(infixPatterns) ++ infixDefaults else $(infixPatterns)

  def getPrefixPattern: String = if ($(includeDefaults)) get(prefixPattern).getOrElse(prefixDefault) else $(prefixPattern)

  def getSuffixPattern: String = if ($(includeDefaults)) get(suffixPattern).getOrElse(suffixDefault) else $(suffixPattern)

  def getTargetPattern: String = $(targetPattern)

  def getIncludeDefaults: Boolean = $(includeDefaults)

  def setIncludeDefaults(value: Boolean): this.type = set(includeDefaults, value)

  setDefault(includeDefaults, true)
  setDefault(targetPattern, "\\S+")
  setDefault(infixPatterns, Array.empty[String])

  /** Check here for explanation on this default pattern */
  private val infixDefaults =  Array(
    "([\\$#]?\\d+(?:[^\\s\\d]{1}\\d+)*)", // Money, Phone number and dates -> http://rubular.com/r/ihCCgJiX4e
    "((?:\\p{L}\\.)+)", // Abbreviations -> http://rubular.com/r/nMf3n0axfQ
    "(\\p{L}+)(n't\\b)", // Weren't -> http://rubular.com/r/coeYJFt8eM
    "(\\p{L}+)('{1}\\p{L}+)", // I'll -> http://rubular.com/r/N84PYwYjQp
    "((?:\\p{L}+[^\\s\\p{L}]{1})+\\p{L}+)", // foo-bar -> http://rubular.com/r/cjit4R6uWd
    "([\\p{L}\\w]+)" // basic word token
  )
  /** These catch everything before and after a word, as a separate token*/
  private val prefixDefault = "\\A([^\\s\\p{L}\\d\\$\\.#]*)"
  private val suffixDefault = "([^\\s\\p{L}\\d]?)([^\\s\\p{L}\\d]*)\\z"

  /** Clears out rules and constructs a new rule for every combination of rules provided */
  /** The strategy is to catch one token per regex group */
  /** User may add its own groups if needs targets to be tokenized separately from the rest */
  lazy private val ruleFactory = {
    val rules = ArrayBuffer.empty[String]
    require(getInfixPatterns.forall(ip => ip.contains("(") && ip.contains(")")),
      "infix patterns must use regex group. Notice each group will result in separate token")
    getInfixPatterns.foreach(ip => {
      val rule = new StringBuilder
      get(prefixPattern).orElse(if (!$(includeDefaults)) None else Some(prefixDefault)).foreach(pp => {
        require(pp.startsWith("\\A"), "prefixPattern must begin with \\A to ensure it is the beginning of the string")
        require(pp.contains("(") && pp.contains(")"), "prefixPattern must contain regex groups. Each group will return in separate token")
        rule.append(pp)
      })
      rule.append(ip)
      get(suffixPattern).orElse(if (!$(includeDefaults)) None else Some(suffixDefault)).foreach(sp => {
        require(sp.endsWith("\\z"), "suffixPattern must end with \\z to ensure it is the end of the string")
        require(sp.contains("(") && sp.contains(")"), "suffixPattern must contain regex groups. Each group will return in separate token")
        rule.append(sp)
      })
      rules.append(rule.toString)
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
      TokenizedSentence(tokens)
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