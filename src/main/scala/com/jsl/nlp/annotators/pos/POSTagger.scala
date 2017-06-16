package com.jsl.nlp.annotators.pos

import com.jsl.nlp.annotators.sbd.SentenceDetector
import com.jsl.nlp.{Annotation, Annotator, Document}

/**
  * Created by Saif Addin on 5/13/2017.
  */
class POSTagger(taggingApproach: POSApproach) extends Annotator {

  private case class SentenceToBeTagged(sentence: String, start: Int, end: Int)

  override val aType: String = POSTagger.aType

  override val requiredAnnotationTypes: Array[String] = Array(SentenceDetector.aType)

  override def annotate(document: Document, annotations: Seq[Annotation]): Seq[Annotation] = {
    val sentences: Array[SentenceToBeTagged] = annotations.collect {
      case sentence: Annotation if sentence.aType == SentenceDetector.aType =>
        SentenceToBeTagged(
          sentence.metadata(SentenceDetector.aType),
          sentence.begin,
          sentence.end
        )
    }.toArray
    taggingApproach.tag(sentences.map(_.sentence))
      .zip(sentences)
      .map{case (taggedWords, sentence) =>
        Annotation(
          POSTagger.aType,
          sentence.start,
          sentence.end,
          taggedWords.map(taggedWord => taggedWord.word -> taggedWord.tag).toMap
        )
      }
  }

}
object POSTagger {
  val aType = "pos"
}