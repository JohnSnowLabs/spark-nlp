package com.jsl.nlp.annotators.sda

import com.jsl.nlp.annotators.Lemmatizer
import com.jsl.nlp.annotators.pos.{POSTagger, TaggedSentence, TaggedWord}
import com.jsl.nlp.annotators.sbd.SentenceDetector
import com.jsl.nlp.{Annotation, Annotator, Document}

/**
  * Created by saif1_000 on 12/06/2017.
  */
class SentimentDetector(sentimentApproach: SentimentApproach) extends Annotator {

  override val aType = SentimentDetector.aType

  override val requiredAnnotationTypes: Array[String] = Array(
    //ToDo: Verify. In this case, order matters. i.e. pos tags must be before lemmatization
    SentenceDetector.aType,
    POSTagger.aType,
    Lemmatizer.aType
  )

  override def annotate(document: Document, annotations: Seq[Annotation]): Seq[Annotation] = {
    val tags = annotations.filter(_.aType == POSTagger.aType)
    val sentences = annotations.filter(_.aType == SentenceDetector.aType)
    val lemmas = annotations.filter(_.aType == Lemmatizer.aType).flatMap(_.metadata).toMap
    sentences.map(sentence => {
      val taggedWords = tags.find(tag => tag.end == sentence.end)
        .getOrElse(throw new Exception("Got a sentence but there were no tags within its range")).metadata
        .map{case (word, tag) => TaggedWord(lemmas.getOrElse(word, word), tag)}.toArray
      TaggedSentence(taggedWords)
    })
    Seq()
  }

}
object SentimentDetector {
  val aType = "sda"
}
