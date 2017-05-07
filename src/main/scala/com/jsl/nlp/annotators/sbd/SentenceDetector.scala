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
        .prepare
        .extract
    sentences.map(s => Annotation(this.aType, s.begin, s.end, Map()))
  }

}
object SentenceDetector {
  val aType = "sbd"
}