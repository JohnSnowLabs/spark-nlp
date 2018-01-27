package com.johnsnowlabs.nlp.annotators

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
  val targetPattern: Param[String] = new Param(this, "targetPattern", "pattern to grab from text as token candidates. Defaults \\S+")
  val infixPatterns: StringArrayParam = new StringArrayParam(this, "infixPattern", "regex patterns that match tokens within a single target. groups identify different sub-tokens. multiple defaults")
  val prefixPattern: Param[String] = new Param[String](this, "prefixPattern", "regex with groups and begins with \\A to match target prefix. Defaults to \\A([^\\s\\w\\$\\.]*)")
  val suffixPattern: Param[String] = new Param[String](this, "suffixPattern", "regex with groups and ends with \\z to match target suffix. Defaults to ([^\\s\\w]?)([^\\s\\w]*)\\z")

  override val annotatorType: AnnotatorType = TOKEN

  /** A Tokenizer could require only for now a SentenceDetector annotator */
  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array[AnnotatorType](DOCUMENT)

  def this() = this(Identifiable.randomUID("REGEX_TOKENIZER"))

  def setTargetPattern(value: String): this.type = set(targetPattern, value)

  def setExtensionPatterns(value: Array[String]): this.type = set(infixPatterns, value)

  def addInfixPattern(value: String): this.type = set(infixPatterns, value +: $(infixPatterns))

  def setPrefixPattern(value: String): this.type = set(prefixPattern, value)

  def setSuffixPattern(value: String): this.type = set(suffixPattern, value)

  def setCompositeTokens(value: Array[String]): this.type = set(compositeTokens, value)

  def getCompositeTokens: Array[String] = $(compositeTokens)

  def getInfixPatterns: Array[String] = $(infixPatterns)

  def getPrefixPattern: String = $(prefixPattern)

  def getSuffixPattern: String = $(suffixPattern)

  def getTargetPattern: String = $(targetPattern)

  setDefault(inputCols, Array(DOCUMENT))

  lazy private val ruleFactory = new RuleFactory(MatchStrategy.MATCH_FIRST)

  /** Clears out rules and constructs a new rule for every combination of rules provided */
  /** The strategy is to catch one token per regex group */
  /** User may add its own groups if needs targets to be tokenized separately from the rest */
  protected def setFactoryRules(): Unit = {
    ruleFactory
      .clearRules()
    val rules = ArrayBuffer.empty[String]
    require($(infixPatterns).nonEmpty)
    require($(infixPatterns).forall(ip => ip.contains("(") && ip.contains(")")),
      "infix patterns must use regex group. Notice each group will result in separate token")
    $(infixPatterns).foreach(ip => {
      val rule = new StringBuilder
      get(prefixPattern).orElse(getDefault(prefixPattern)).foreach(pp => {
        require(pp.startsWith("\\A"), "prefixPattern must begin with \\A to ensure it is the beginning of the string")
        require(pp.contains("(") && pp.contains(")"), "prefixPattern must contain regex groups. Each group will return in separate token")
        rule.append(pp)
      })
      rule.append(ip)
      get(suffixPattern).orElse(getDefault(suffixPattern)).foreach(sp => {
        require(sp.endsWith("\\z"), "suffixPattern must end with \\z to ensure it is the end of the string")
        require(sp.contains("(") && sp.contains(")"), "suffixPattern must contain regex groups. Each group will return in separate token")
        rule.append(sp)
      })
      rules.append(rule.toString)
    })
    rules.foreach(rule => ruleFactory.addRule(rule.r, rule))
  }

  /** Check here for explanation on this default pattern */
  setDefault(infixPatterns, Array(
    "((?:\\w+\\.)+)", // http://rubular.com/r/cRBtGuLlF6
    "(\\w+)(n't\\b)", // http://rubular.com/r/coeYJFt8eM
    "(\\w+)('{1}\\w+)", // http://rubular.com/r/N84PYwYjQp
    "((?:\\w+[^\\s\\w]{1})+\\w+)", // http://rubular.com/r/wOvQcey9e3
    "(\\w+)" // basic word token
  ))
  /** These catch everything before and after a word, as a separate token*/
  setDefault(prefixPattern, "\\A([^\\s\\w\\$\\.]*)")
  setDefault(suffixPattern, "([^\\s\\w]?)([^\\s\\w]*)\\z")
  setDefault(targetPattern, "\\S+")

  setFactoryRules()

  override def beforeAnnotate(): Unit = {
    setFactoryRules()
  }

  private val PROTECT_STR = "ↈ"

  def tag(sentences: Seq[Sentence]): Seq[TokenizedSentence] = {
    sentences.map{text =>
      /** Step 1, protect exception words from being broken*/
      var protected_text = text.content
      if (get(compositeTokens).isDefined) {
        $(compositeTokens).foreach(tokenException =>
          protected_text = protected_text.replaceAll(
            tokenException,
            tokenException.replaceAll("[^(?:" + $(targetPattern) + ")]", PROTECT_STR)
          )
        )
      }
      /** Step 2, Return protected exception tokens back into text and move on*/
      val tokens = $(targetPattern).r.findAllMatchIn(protected_text).flatMap { candidate =>
        if (get(compositeTokens).isDefined && candidate.matched.contains(PROTECT_STR)) {
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
    Tokenized.pack(tokenized)
  }
}

object Tokenizer extends DefaultParamsReadable[Tokenizer]