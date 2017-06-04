package com.jsl.nlp.annotators

import com.jsl.nlp.{Annotation, Annotator, Document}

/**
  * Created by Saif Addin on 6/3/2017.
  */
class DateMatcher extends Annotator {

  case class MatchedDate(start: Int, end: Int, content: String)

  override val aType: String = DateMatcher.aType

  override val requiredAnnotationTypes: Array[String] = Array()

  override def annotate(document: Document, annotations: Seq[Annotation]): Seq[Annotation] = {
    findDate(document.text).map(matchedDate => Annotation(
      DateMatcher.aType,
      matchedDate.start,
      matchedDate.end,
      Map(DateMatcher.aType -> matchedDate.content)
    ))
  }

  private def findDate(text: String): Array[MatchedDate] = {

  }

}
object DateMatcher {
  val aType: String = "date"
}