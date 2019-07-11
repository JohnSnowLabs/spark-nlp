package com.johnsnowlabs.nlp.annotators

import java.util.regex.Pattern

import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.pretrained.ResourceDownloader
import com.johnsnowlabs.nlp.serialization.StructFeature
import com.johnsnowlabs.nlp.util.regex.{MatchStrategy, RuleFactory}
import org.apache.spark.ml.param.{BooleanParam, Param, StringArrayParam}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, ParamsAndFeaturesReadable}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

import scala.collection.mutable.ArrayBuffer

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

  private val PROTECT_CHAR = "ↈ"
  private val BREAK_CHAR = "ↇ"

  private lazy val BREAK_PATTERN = "[^(?:" + $(targetPattern) + ")" + PROTECT_CHAR + "]"
  private lazy val SPLIT_PATTERN = "[^" + BREAK_CHAR + "]+"

  private def casedMatchExists (candidateMatched: String): Boolean =
    if ($(caseSensitiveExceptions))
      $(exceptions).contains(candidateMatched)
    else
      $(exceptions).map(_.toLowerCase).contains(candidateMatched.toLowerCase)

  def tag(sentences: Seq[Sentence]): Seq[TokenizedSentence] = {
    sentences.map{text =>
      /** Step 1, define breaks from non breaks */
      val protectedText = {
        get(exceptions).map(_.foldRight(text.content)((exceptionToken, currentText) => {
          val casedExceptionToken = if ($(caseSensitiveExceptions)) exceptionToken else "(?i)"+exceptionToken
          currentText.replaceAll(
            casedExceptionToken,
            exceptionToken.replaceAll(BREAK_PATTERN, PROTECT_CHAR)
          )
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
        $$(rules).findMatchFirstOnly(candidate.matched).map {m =>
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

trait PretrainedTokenizer {
  def pretrained(name: String = "token_rules", lang: String = "en", remoteLoc: String = ResourceDownloader.publicLoc): TokenizerModel = {
    ResourceDownloader.downloadModel(TokenizerModel, name, Option(lang), remoteLoc)
  }
}

object TokenizerModel extends ParamsAndFeaturesReadable[TokenizerModel] with PretrainedTokenizer