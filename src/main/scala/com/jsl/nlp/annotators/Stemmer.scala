package com.jsl.nlp.annotators

import com.jsl.nlp.{Annotation, Annotator, Document}
import opennlp.tools.stemmer.PorterStemmer
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

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

  def this() = this(Identifiable.randomUID(Stemmer.annotatorType))

  /** one-to-one stem annotation that returns single hard-stem per token */
  override def annotate(
                         document: Document, annotations: Seq[Annotation]
  ): Seq[Annotation] =
    annotations.collect {
      case tokenAnnotation: Annotation if tokenAnnotation.annotatorType == RegexTokenizer.annotatorType =>
        val token = document.text.substring(tokenAnnotation.begin, tokenAnnotation.end)
        val stem = Stemmer.stemmer.stem(token)
        Annotation(annotatorType, tokenAnnotation.begin, tokenAnnotation.end, Map(token -> stem))
    }

}

object Stemmer extends DefaultParamsReadable[Stemmer] {
  val annotatorType = "stem"
  private val stemmer = new PorterStemmer()
}