package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.serialization.StructFeature
import com.johnsnowlabs.nlp.util.regex.RuleFactory
import org.apache.spark.ml.param.{BooleanParam, IntParam, Param, StringArrayParam}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, HasPretrained, ParamsAndFeaturesReadable}
import org.apache.spark.ml.util.Identifiable

/**
  * Tokenizes raw text into word pieces, tokens.
  * @param uid required uid for storing annotator to disk
  * @@ pattern: RegexPattern to split phrases into tokens
  */
class TokenizerModel(override val uid: String) extends AnnotatorModel[TokenizerModel] {

  import com.johnsnowlabs.nlp.AnnotatorType._

  val rules: StructFeature[RuleFactory] = new StructFeature[RuleFactory](this, "rules")
  val exceptions: StringArrayParam = new StringArrayParam(this, "exceptions", "Words that won't be affected by tokenization rules")
  val caseSensitiveExceptions: BooleanParam = new BooleanParam(this, "caseSensitiveExceptions", "Whether to care for case sensitiveness in exceptions")
  val targetPattern: Param[String] = new Param(this, "targetPattern", "pattern to grab from text as token candidates. Defaults \\S+")
  val minLength = new IntParam(this, "minLength", "Set the minimum allowed legth for each token")
  val maxLength = new IntParam(this, "maxLength", "Set the maximum allowed legth for each token")
  val splitChars: StringArrayParam = new StringArrayParam(this, "splitChars", "character list used to separate from the inside of tokens")

  setDefault(
    targetPattern -> "\\S+",
    caseSensitiveExceptions -> true
  )

  override val outputAnnotatorType: AnnotatorType = TOKEN

  /** A Tokenizer could require only for now a SentenceDetector annotator */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array[AnnotatorType](DOCUMENT)

  def this() = this(Identifiable.randomUID("REGEX_TOKENIZER"))

  def setTargetPattern(value: String): this.type = set(targetPattern, value)
  def getTargetPattern: String = $(targetPattern)

  def setExceptions(value: Array[String]): this.type = set(exceptions, value)
  def getExceptions: Array[String] = $(exceptions)

  def setRules(ruleFactory: RuleFactory): this.type = set(rules, ruleFactory)

  def setCaseSensitiveExceptions(value: Boolean): this.type = set(caseSensitiveExceptions, value)
  def getCaseSensitiveExceptions(value: Boolean): Boolean = $(caseSensitiveExceptions)

  def setMinLength(value: Int): this.type = set(minLength, value)
  def getMinLength(value: Int): Int = $(minLength)

  def setMaxLength(value: Int): this.type = set(maxLength, value)
  def getMaxLength(value: Int): Int = $(maxLength)

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

  private val PROTECT_CHAR = "ↈ"
  private val BREAK_CHAR = "ↇ"

  private lazy val BREAK_PATTERN = "[^(?:" + $(targetPattern) + ")" + PROTECT_CHAR + "]"
  private lazy val SPLIT_PATTERN = "[^" + BREAK_CHAR + "]+"

  private def casedMatchExists (candidateMatched: String): Boolean =
    if ($(caseSensitiveExceptions))
      $(exceptions).exists(e => e.r.findFirstIn(candidateMatched).isDefined)
    else
      $(exceptions).exists(e => ("(?i)"+e).r.findFirstIn(candidateMatched).isDefined)

  def tag(sentences: Seq[Sentence]): Seq[TokenizedSentence] = {
    sentences.map{text =>
      /** Step 1, define breaks from non breaks */
      val protectedText = {
        get(exceptions).map(_.foldRight(text.content)((exceptionToken, currentText) => {
          val casedExceptionPattern = if ($(caseSensitiveExceptions)) exceptionToken else "(?i)"+exceptionToken
          casedExceptionPattern.r.replaceAllIn(currentText, m => m.matched.replaceAll(BREAK_PATTERN, PROTECT_CHAR))
        })).getOrElse(text.content).replaceAll(BREAK_PATTERN, BREAK_CHAR)
      }
      /** Step 2, Return protected tokens back into text and move on*/
      val tokens = SPLIT_PATTERN.r.findAllMatchIn(protectedText).flatMap { candidate =>
        if (get(exceptions).isDefined &&
          (
            candidate.matched.contains(PROTECT_CHAR) ||
              casedMatchExists(candidate.matched)
            )) {
          /** Put back character and move on */
          Seq(IndexedToken(
            text.content.slice(text.start + candidate.start, text.start + candidate.end),
            text.start + candidate.start,
            text.start + candidate.end - 1
          ))
        } else {
          /** Step 3, If no exception found, find candidates through the possible general rule patterns*/
          val rr = $$(rules).findMatchFirstOnly(candidate.matched).map {m =>
            var curPos = m.content.start
            (1 to m.content.groupCount)
              .flatMap (i => {
                val target = m.content.group(i)
                if (target.nonEmpty && isSet(splitChars) && $(splitChars).exists(target.contains)) {
                  try {
                    val strs = target.split($(splitChars).mkString("|"))
                    strs.map(str =>
                      try {
                        IndexedToken(
                          str,
                          text.start + candidate.start + curPos,
                          text.start + candidate.start + curPos + str.length - 1
                        )
                      } finally {
                        curPos += str.length + 1
                      }
                    )
                  } finally {
                    curPos -= 1
                  }
                } else {
                  val it = IndexedToken(
                    target,
                    text.start + candidate.start + curPos,
                    text.start + candidate.start + curPos + target.length - 1
                  )
                  curPos += target.length
                  Seq(it)
                }
              })
            /** Step 4, If rules didn't match, return whatever candidate we have and leave it as is*/
          }.getOrElse(Seq(IndexedToken(
            candidate.matched,
            text.start + candidate.start,
            text.start + candidate.end - 1
          )))
          rr
        }
      }.filter(t => t.token.nonEmpty && t.token.length >= $(minLength) && get(maxLength).forall(m => t.token.length <= m)).toArray
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

trait ReadablePretrainedTokenizer extends ParamsAndFeaturesReadable[TokenizerModel] with HasPretrained[TokenizerModel] {
  override val defaultModelName = Some("token_rules")
  /** Java compliant-overrides */
  override def pretrained(): TokenizerModel = super.pretrained()
  override def pretrained(name: String): TokenizerModel = super.pretrained(name)
  override def pretrained(name: String, lang: String): TokenizerModel = super.pretrained(name, lang)
  override def pretrained(name: String, lang: String, remoteLoc: String): TokenizerModel = super.pretrained(name, lang, remoteLoc)
}

object TokenizerModel extends ReadablePretrainedTokenizer