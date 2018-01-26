package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.util.regex.{MatchStrategy, RuleFactory}
import org.apache.spark.ml.param.{Param, StringArrayParam}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, AnnotatorType}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

import scala.util.matching.Regex

/**
  * Tokenizes raw text into word pieces, tokens.
  * @param uid required uid for storing annotator to disk
  * @@ pattern: RegexPattern to split phrases into tokens
  */
class Tokenizer(override val uid: String) extends AnnotatorModel[Tokenizer] {

  import com.johnsnowlabs.nlp.AnnotatorType._

  val wordPattern: Param[String] = new Param(this, "wordPattern", "this is the base word pattern. Defaults \\w+")
  val extensionPattern: StringArrayParam = new StringArrayParam(this, "infixPattern", "infix patterns allow for word exceptions that count as single token. E.g. U.S.A. Defaults ")
  val prefixPattern: StringArrayParam = new StringArrayParam(this, "prefixPattern", "this is the token pattern")
  val suffixPattern: StringArrayParam = new StringArrayParam(this, "suffixPattern", "this is the token pattern")

  override val annotatorType: AnnotatorType = TOKEN

  /** A Tokenizer could require only for now a SentenceDetector annotator */
  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array[AnnotatorType](DOCUMENT)

  def this() = this(Identifiable.randomUID("REGEX_TOKENIZER"))

  def setWordPattern(value: String): this.type = set(wordPattern, value)

  def setExtensionPattern(value: Array[String]): this.type = set(extensionPattern, value)

  def addExtensionPattern(value: String): this.type = set(extensionPattern, $(extensionPattern) :+ value)

  def setPrefixPattern(value: Array[String]): this.type = set(prefixPattern, value)

  def addPrefixPattern(value: String): this.type = set(prefixPattern, $(prefixPattern) :+ value)

  def setSuffixPattern(value: Array[String]): this.type = set(suffixPattern, value)

  def addSuffixPattern(value: String): this.type = set(suffixPattern, $(suffixPattern) :+ value)

  def getWordPattern: String = $(wordPattern)

  def getInfixPattern: Array[String] = $(extensionPattern)

  def getPrefixPattern: Array[String] = $(prefixPattern)

  def getSuffixPattern: Array[String] = $(suffixPattern)

  setDefault(inputCols, Array(DOCUMENT))

  setDefault(wordPattern, "\\w+")
  setDefault(extensionPattern, Array("\\.(?:\\w{1}\\.)+|(?:\\-\\w+)*"))
  setDefault(prefixPattern, Array("([^\\s\\w]?)"))
  setDefault(suffixPattern, Array("([^\\s\\w]?)([^\\s\\w]*)"))

  val ruleFactory = new RuleFactory(MatchStrategy.MATCH_ALL)

  override def beforeAnnotate(): Unit = {
    /** Clears out rules and constructs a new rule for every combination of rules provided */
    /** The strategy is to catch one token per regex group */
    /** User may add its own groups if needs targets to be tokenized separately from the rest */
    /** "([^\s\w]?)(\w+(?:\.(?:\w{1}\.)+|(?:\-\w+)*)?)([^\s\w]?)([\s\w]*)" */
    /** */
    ruleFactory
      .clearRules()
    $(prefixPattern).foreach(pp => $(suffixPattern).foreach (sp => $(extensionPattern).foreach(ep => {
      ruleFactory.addRule(
        (pp + "(" + $(wordPattern) + "(?:" + ep + ")?" + ")" + sp).r,
        "tokenizer construction pattern"
      )
    })))
  }

  def tag(sentences: Seq[Sentence]): Seq[TokenizedSentence] = {
    sentences.map{text =>
      val tokens = ruleFactory.findMatch(text.content).flatMap { m =>
        (1 to m.content.groupCount)
          .map (i => IndexedToken(m.content.group(i), text.begin + m.content.start, text.begin + m.content.end - 1))
      }.filter(t => t.token.nonEmpty).toArray
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