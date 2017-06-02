package com.jsl.nlp.annotators.sbd

import com.jsl.nlp.{Annotation, Annotator, Document}

/**
  * Created by Saif Addin on 5/5/2017.
  */
class SentenceDetector(detectionApproach: SBDApproach) extends Annotator {

  override val aType: String = SentenceDetector.aType

  override val requiredAnnotationTypes: Seq[String] = Seq()

  override def annotate(document: Document, annotations: Seq[Annotation]): Seq[Annotation] = {
    val sentences: Seq[Sentence] =
      detectionApproach
        .setContent(document.text)
        .prepare
        .extract
    sentences.map(sentence => Annotation(
      this.aType,
      sentence.begin,
      sentence.end,
      Map[String, String](this.aType -> sentence.content)
    ))
  }

}
object SentenceDetector {
  val aType = "sbd"
}