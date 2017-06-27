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

/**
  * Part of speech tagger that might use different approaches
  * @param uid Internal constructor requirement for serialization of params
  * @@model: representation of a POS Tagger approach
  */
class POSTagger(override val uid: String) extends Annotator {

  /** Internal structure for target sentences holding their range information which is used for annotation */
  private case class SentenceToBeTagged(tokenizedSentence: TokenizedSentence, start: Int, end: Int)

  val model: AnnotatorParam[POSApproach, SerializedPerceptronApproach] =
    new AnnotatorParam[POSApproach, SerializedPerceptronApproach](this, "POS Model", "POS Tagging approach")

  override val annotatorType: String = POSTagger.annotatorType

  /** POS tagging requires sentence in order to provide a context for disambiguation
    * Tokens are required to identify POS Tags words by word in a sentence boundary
    */
  override var requiredAnnotatorTypes: Array[String] = Array(
    SentenceDetector.annotatorType,
    RegexTokenizer.annotatorType
  )

  def this() = this(Identifiable.randomUID(POSTagger.annotatorType))

  def getModel: POSApproach = $(model)

  def setModel(targetModel: POSApproach): this.type = set(model, targetModel)

  /** One to one annotation standing from the Tokens perspective, to give each word a corresponding Tag */
  override def annotate(document: Document, annotations: Seq[Annotation]): Seq[Annotation] = {
    val sentences: Array[SentenceToBeTagged] = annotations.collect {
      case sentence: Annotation if sentence.annotatorType == SentenceDetector.annotatorType =>
        /** Creates a sentence bounded by tokens in a sentence */
        val tokenizedSentence = TokenizedSentence(
          annotations.filter(annotation =>
            annotation.annotatorType == RegexTokenizer.annotatorType &&
            annotation.end <= sentence.end
          ).map(_.metadata(RegexTokenizer.annotatorType)).toArray
        )
        /** Tags the sentence in a token manner while holding sentence bounds */
        SentenceToBeTagged(
          tokenizedSentence,
          sentence.begin,
          sentence.end
        )
    }.toArray
    /** Tags the sentence tokens while holding the sentence for annotation*/
    getModel.tag(sentences.map(_.tokenizedSentence))
      .zip(sentences)
      .map{case (taggedWords, sentence) =>
        Annotation(
          POSTagger.annotatorType,
          sentence.start,
          sentence.end,
          taggedWords.mapWords
        )
      }
  }

}
object POSTagger extends DefaultParamsReadable[POSTagger] {
  val annotatorType = "pos"
}