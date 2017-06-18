package com.jsl.nlp.annotators.sda

import com.jsl.nlp.annotators.common.{TaggedSentence, TaggedWord}
import com.jsl.nlp.annotators.{Lemmatizer, RegexTokenizer}
import com.jsl.nlp.annotators.pos.POSTagger
import com.jsl.nlp.annotators.sbd.SentenceDetector
import com.jsl.nlp.{Annotation, Annotator, Document}

/**
  * Created by saif1_000 on 12/06/2017.
  */
class SentimentDetector(
                        sentimentApproach: SentimentApproach
                       ) extends Annotator {

  override val aType = SentimentDetector.aType

  //ToDo: Verify. In this case, order matters. i.e. pos tags must be before lemmatization
  override val requiredAnnotationTypes: Array[String] = {
    var requiredAnnotations = Array(
      RegexTokenizer.aType,
      SentenceDetector.aType
    )
    if (sentimentApproach.requiresPOS)
      requiredAnnotations = requiredAnnotations :+ POSTagger.aType
    if (sentimentApproach.requiresLemmas)
      requiredAnnotations = requiredAnnotations :+ Lemmatizer.aType
    requiredAnnotations
  }

  override def annotate(document: Document, annotations: Seq[Annotation]): Seq[Annotation] = {
    val tokens = annotations.filter(_.aType == RegexTokenizer.aType)
    val sentences = annotations.filter(_.aType == SentenceDetector.aType)
    val tags = annotations.filter(_.aType == POSTagger.aType)
    val lemmas = annotations.filter(_.aType == Lemmatizer.aType).flatMap(_.metadata).toMap
    val taggedSentences = sentences.map(sentence => {
      val taggedWords = tags.find(tag => tag.end == sentence.end).map(_.metadata)
        .getOrElse(tokens.filter(_.end <= sentence.end).flatMap(_.metadata.values))
        .map {
          case (word: String, tag: String) => TaggedWord(lemmas.getOrElse(word, word), tag)
          case word: String => TaggedWord(word, "?NOTAG?")
        }.toArray
      TaggedSentence(taggedWords)
    }).toArray
    val score = sentimentApproach.score(taggedSentences)
    Seq(Annotation(
      SentimentDetector.aType,
      0,
      0,
      Map(SentimentDetector.aType -> score.toString)
    ))
  }

}
object SentimentDetector {
  val aType = "sda"
}
