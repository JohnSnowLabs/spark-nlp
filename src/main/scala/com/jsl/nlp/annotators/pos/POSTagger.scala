package com.jsl.nlp.annotators.pos

import com.jsl.nlp.annotators.RegexTokenizer
import com.jsl.nlp.annotators.common.TokenizedSentence
import com.jsl.nlp.annotators.sbd.SentenceDetector
import com.jsl.nlp.annotators.param.AnnotatorParam
import com.jsl.nlp.annotators.pos.perceptron.SerializedPerceptronApproach
import com.jsl.nlp.{Annotation, Annotator, Document}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

/**
  * Created by Saif Addin on 5/13/2017.
  */
class POSTagger(override val uid: String) extends Annotator {

  private case class SentenceToBeTagged(tokenizedSentence: TokenizedSentence, start: Int, end: Int)

  val model: AnnotatorParam[POSApproach, SerializedPerceptronApproach] =
    new AnnotatorParam[POSApproach, SerializedPerceptronApproach](this, "POS Model", "POS Tagging approach")

  override val annotatorType: String = POSTagger.aType

  override var requiredAnnotatorTypes: Array[String] = Array(
    SentenceDetector.aType,
    RegexTokenizer.annotatorType
  )

  def this() = this(Identifiable.randomUID(POSTagger.aType))

  def getModel: POSApproach = $(model)

  def setModel(targetModel: POSApproach): this.type = set(model, targetModel)

  override def annotate(document: Document, annotations: Seq[Annotation]): Seq[Annotation] = {
    val sentences: Array[SentenceToBeTagged] = annotations.collect {
      case sentence: Annotation if sentence.annotatorType == SentenceDetector.aType =>
        val tokenizedSentence = TokenizedSentence(
          annotations.filter(annotation =>
            annotation.annotatorType == RegexTokenizer.annotatorType &&
            annotation.end <= sentence.end
          ).map(_.metadata(RegexTokenizer.annotatorType)).toArray
        )
        SentenceToBeTagged(
          tokenizedSentence,
          sentence.begin,
          sentence.end
        )
    }.toArray
    getModel.tag(sentences.map(_.tokenizedSentence))
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
object POSTagger extends DefaultParamsReadable[POSTagger] {
  val aType = "pos"
}