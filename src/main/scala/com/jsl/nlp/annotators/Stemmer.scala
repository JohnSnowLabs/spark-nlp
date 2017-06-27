package com.jsl.nlp.annotators

import com.jsl.nlp.{Annotation, Annotator, Document}
import opennlp.tools.stemmer.PorterStemmer
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

/**
  * Created by alext on 10/23/16.
  */
class Stemmer(override val uid: String) extends Annotator {

  override val aType: String = Stemmer.aType

  override var requiredAnnotationTypes = Array(RegexTokenizer.aType)

  def this() = this(Identifiable.randomUID(Stemmer.aType))

  override def annotate(
                         document: Document, annotations: Seq[Annotation]
  ): Seq[Annotation] =
    annotations.collect {
      case tokenAnnotation: Annotation if tokenAnnotation.aType == RegexTokenizer.aType =>
        val token = document.text.substring(tokenAnnotation.begin, tokenAnnotation.end)
        val stem = Stemmer.stemmer.stem(token)
        Annotation(aType, tokenAnnotation.begin, tokenAnnotation.end, Map(token -> stem))
    }

}

object Stemmer extends DefaultParamsReadable[Stemmer] {
  val aType = "stem"
  private val stemmer = new PorterStemmer()
}