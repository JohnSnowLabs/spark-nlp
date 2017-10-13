package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.ml.param.Param
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, AnnotatorType, DocumentAssembler}
import com.typesafe.config.{Config, ConfigFactory}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

import scala.util.matching.Regex

/**
  * Tokenizes raw text into word pieces, tokens.
  * @param uid required uid for storing annotator to disk
  * @@ pattern: RegexPattern to split phrases into tokens
  */
class RegexTokenizer(override val uid: String) extends AnnotatorModel[RegexTokenizer] {

  import com.johnsnowlabs.nlp.AnnotatorType._

  val pattern: Param[String] = new Param(this, "pattern", "this is the token pattern")

  lazy val regex: Regex = $(pattern).r

  override val annotatorType: AnnotatorType = TOKEN

  def this() = this(Identifiable.randomUID("REGEX_TOKENIZER"))

  def setPattern(value: String): this.type = set(pattern, value)

  def getPattern: String = $(pattern)

  setDefault(inputCols, Array(DOCUMENT))

  /** A RegexTokenizer could require only for now a SentenceDetectorModel annotator */
  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array[AnnotatorType](DOCUMENT)

  setDefault(pattern, "\\S+")

  /** one to many annotation */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val sentenceLength = annotations.length
    var sentenceIndex = if (sentenceLength <= 1) -1 else 0
    annotations.flatMap(text => {
      sentenceIndex += 1
      regex.findAllMatchIn(text.metadata(AnnotatorType.DOCUMENT)).map { m =>
        Annotation(
          annotatorType,
          text.begin + m.start,
          text.begin + m.end - 1,
          Map(
            annotatorType -> m.matched,
            "sentence" -> sentenceIndex.toString
          )
        )
      }
    })
  }

}
object RegexTokenizer extends DefaultParamsReadable[RegexTokenizer]