package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.annotators.common._
import org.apache.spark.ml.param.Param
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

  val pattern: Param[String] = new Param(this, "pattern", "this is the token pattern")

  lazy val regex: Regex = $(pattern).r

  override val annotatorType: AnnotatorType = TOKEN

  def this() = this(Identifiable.randomUID("REGEX_TOKENIZER"))

  def setPattern(value: String): this.type = set(pattern, value)

  def addPattern(value: String)

  def setPrefixPattern(value: String)

  def setSuffixPattern(value: String)

  def addPrefixPattern(value: String)

  def addSuffixPattern(value: String)

  def getPattern: String = $(pattern)

  def getPrefixPattern: String

  def getSuffixPattern: String

  setDefault(inputCols, Array(DOCUMENT))

  /** A RegexTokenizer could require only for now a SentenceDetectorModel annotator */
  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array[AnnotatorType](DOCUMENT)

  setDefault(pattern, "([^\\s\\w]?)(\\w+(?:\\.\\w{1}\\.|(?:\\-\\w+)*)?)([^\\s\\w]?)")

  def tag(sentences: Seq[Sentence]): Seq[TokenizedSentence] = {
    sentences.map{text =>
      val tokens = regex.findAllMatchIn(text.content).flatMap { m =>
        (1 to m.groupCount).map (i => IndexedToken(m.group(i), text.begin + m.start, text.begin + m.end - 1))
      }.toArray
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