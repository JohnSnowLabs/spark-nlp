package com.jsl.nlp.annotators

import com.jsl.nlp.{Annotation, Annotator, Document}
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable
import org.tartarus.snowball.SnowballStemmer

/**
  * Created by alext on 10/23/16.
  */

/**
  * Hard stemming of words for cut-of into standard word references
  * @param uid internal uid element for storing annotator into disk
  */
class Stemmer(override val uid: String) extends Annotator {

  override val annotatorType: String = Stemmer.annotatorType

  override var requiredAnnotatorTypes = Array(RegexTokenizer.annotatorType)

  private val stemmer: (String => String) = Stemmer.getStemmer($(algorithm))

  val algorithm: Param[String] = new Param(this, "language", "this is the language of the text")
  setDefault(algorithm, "english")

  def setPattern(value: String): Stemmer = set(algorithm, value)

  def getPattern: String = $(algorithm)

  def this() = this(Identifiable.randomUID(Stemmer.annotatorType))

  /** one-to-one stem annotation that returns single hard-stem per token */
  override def annotate(
                         document: Document, annotations: Seq[Annotation]
  ): Seq[Annotation] =
    annotations.collect {
      case tokenAnnotation: Annotation if tokenAnnotation.annotatorType == RegexTokenizer.annotatorType =>
        val stem = stemmer(document.text.substring(tokenAnnotation.begin, tokenAnnotation.end))
        Annotation(annotatorType, tokenAnnotation.begin, tokenAnnotation.end, Map(Stemmer.annotatorType -> stem))
    }

}

object Stemmer {
  val annotatorType = "stem"

  private def getStemmer(algorithm: String = "english"): (String => String) = {
    val stemmerClass = if (algorithm.indexOf('.') == -1) {
      Class.forName(s"org.tartarus.snowball.ext.${algorithm.toLowerCase()}Stemmer")
    } else {
      Class.forName(algorithm)
    }
    (token: String) => {
      val stemmer = stemmerClass.newInstance().asInstanceOf[SnowballStemmer]
      stemmer.setCurrent(token)
      stemmer.stem()
      stemmer.getCurrent
    }
  }
}