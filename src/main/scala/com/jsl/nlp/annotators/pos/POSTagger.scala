package com.jsl.nlp.annotators.pos

import com.jsl.nlp.annotators.RegexTokenizer
import com.jsl.nlp.annotators.common.TokenizedSentence
import com.jsl.nlp.annotators.sbd.SentenceDetector
import com.jsl.nlp.{Annotation, Annotator, Document}

/**
  * Created by Saif Addin on 5/13/2017.
  */
class POSTagger(taggingApproach: POSApproach) extends Annotator {

  private case class SentenceToBeTagged(tokenizedSentence: TokenizedSentence, start: Int, end: Int)

  override val aType: String = POSTagger.aType

  override val requiredAnnotationTypes: Array[String] = Array(
    SentenceDetector.aType,
    RegexTokenizer.aType
  )

  override def annotate(document: Document, annotations: Seq[Annotation]): Seq[Annotation] = {
    val sentences: Array[SentenceToBeTagged] = annotations.collect {
      case sentence: Annotation if sentence.aType == SentenceDetector.aType =>
        val tokenizedSentence = TokenizedSentence(
          annotations.filter(annotation =>
            annotation.aType == RegexTokenizer.aType &&
            annotation.end <= sentence.end
          ).map(_.metadata(RegexTokenizer.aType)).toArray
        )
        SentenceToBeTagged(
          tokenizedSentence,
          sentence.begin,
          sentence.end
        )
    }.toArray
    taggingApproach.tag(sentences.map(_.tokenizedSentence))
      .zip(sentences)
      .map{case (taggedWords, sentence) =>
        Annotation(
          POSTagger.aType,
          sentence.start,
          sentence.end,
          taggedWords.mapWords
        )
      }
  }

}
object POSTagger {
  val aType = "pos"
}