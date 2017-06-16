package com.jsl.nlp.annotators.sda

import com.jsl.nlp.annotators.pos.POSTagger
import com.jsl.nlp.annotators.sbd.SentenceDetector
import com.jsl.nlp.{Annotation, Annotator, Document}

/**
  * Created by saif1_000 on 12/06/2017.
  */
class SentimentDetector(sentimentApproach: SentimentApproach) extends Annotator {

  override val aType = SentimentDetector.aType

  override val requiredAnnotationTypes: Array[String] = Array(
    SentenceDetector.aType, // Redundant due to transitivity
    POSTagger.aType
  )

  override def annotate(document: Document, annotations: Seq[Annotation]): Seq[Annotation] = {

  }

}
object SentimentDetector {
  val aType = "sda"
}
