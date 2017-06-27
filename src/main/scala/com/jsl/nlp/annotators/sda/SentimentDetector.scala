package com.jsl.nlp.annotators.sda

import com.jsl.nlp.annotators.common.{TaggedSentence, TaggedWord}
import com.jsl.nlp.annotators.param.AnnotatorParam
import com.jsl.nlp.annotators.{Lemmatizer, RegexTokenizer}
import com.jsl.nlp.annotators.pos.POSTagger
import com.jsl.nlp.annotators.sbd.SentenceDetector
import com.jsl.nlp.annotators.sda.pragmatic.SerializedScorerApproach
import com.jsl.nlp.{Annotation, Annotator, Document}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

/**
  * Created by saif on 12/06/2017.
  */
class SentimentDetector(override val uid: String) extends Annotator {

  val model: AnnotatorParam[SentimentApproach, SerializedScorerApproach] =
    new AnnotatorParam[SentimentApproach, SerializedScorerApproach](
      this,
      "Sentiment detection model",
      "Approach to translate into expressed sentiment"
    )

  override val annotatorType: String = SentimentDetector.aType

  //ToDo: Verify. In this case, order matters. i.e. pos tags must be before lemmatization
  override var requiredAnnotatorTypes: Array[String] = Array(
    RegexTokenizer.annotatorType,
    SentenceDetector.aType
  )

  def this() = this(Identifiable.randomUID(SentimentDetector.aType))

  def getModel: SentimentApproach = $(model)

  def setModel(targetModel: SentimentApproach): this.type = {
    if (targetModel.requiresPOS) requiredAnnotatorTypes = requiredAnnotatorTypes :+ POSTagger.aType
    if (targetModel.requiresLemmas) requiredAnnotatorTypes = requiredAnnotatorTypes :+ Lemmatizer.annotatorType
    set(model, targetModel)
  }

  override def annotate(document: Document, annotations: Seq[Annotation]): Seq[Annotation] = {
    val tokens = annotations.filter(_.annotatorType == RegexTokenizer.annotatorType)
    val sentences = annotations.filter(_.annotatorType == SentenceDetector.aType)
    val tags = annotations.filter(_.annotatorType == POSTagger.aType)
    val lemmas = annotations.filter(_.annotatorType == Lemmatizer.annotatorType).flatMap(_.metadata).toMap
    val taggedSentences = sentences.map(sentence => {
      val taggedWords = tags.find(tag => tag.end == sentence.end).map(_.metadata)
        .getOrElse(tokens.filter(_.end <= sentence.end).flatMap(_.metadata.values))
        .map {
          case (word: String, tag: String) => TaggedWord(lemmas.getOrElse(word, word), tag)
          case word: String => TaggedWord(word, "?NOTAG?")
        }.toArray
      TaggedSentence(taggedWords)
    }).toArray
    val score = getModel.score(taggedSentences)
    Seq(Annotation(
      SentimentDetector.aType,
      0,
      0,
      Map(SentimentDetector.aType -> score.toString)
    ))
  }

}
object SentimentDetector extends DefaultParamsReadable[SentimentDetector] {
  val aType = "sda"
}
