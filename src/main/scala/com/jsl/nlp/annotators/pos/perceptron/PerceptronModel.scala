package com.jsl.nlp.annotators.pos.perceptron

import com.jsl.nlp.annotators.common.{IndexedTaggedWord, IndexedToken, TaggedSentence, TokenizedSentence}
import com.jsl.nlp.annotators.param.AnnotatorParam
import com.jsl.nlp.{Annotation, AnnotatorModel}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

/**
  * Part of speech tagger that might use different approaches
  * @param uid Internal constructor requirement for serialization of params
  * @@model: representation of a POS Tagger approach
  */
class PerceptronModel(override val uid: String) extends AnnotatorModel[PerceptronModel] {

  import PerceptronApproach._
  import com.jsl.nlp.AnnotatorType._

  /** Internal structure for target sentences holding their range information which is used for annotation */
  private case class SentenceToBeTagged(tokenizedSentence: TokenizedSentence, start: Int, end: Int)

  val model: AnnotatorParam[AveragedPerceptron, SerializedPerceptronModel] =
    new AnnotatorParam[AveragedPerceptron, SerializedPerceptronModel](this, "POS Model", "POS Tagging approach")

  override val annotatorType: AnnotatorType = POS

  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN, DOCUMENT)

  /**
    * Tags a group of sentences into POS tagged sentences
    * The logic here is to create a sentence context, run through every word and evaluate its context
    * Based on how frequent a context appears around a word, such context is given a score which is used to predict
    * Some words are marked as non ambiguous from the beginning
    * @param tokenizedSentences Sentence in the form of single word tokens
    * @return A list of sentences which have every word tagged
    */
  def tag(tokenizedSentences: Array[TokenizedSentence]): Array[TaggedSentence] = {
    logger.debug(s"PREDICTION: Tagging:\nSENT: <<${tokenizedSentences.map(_.condense).mkString(">>\nSENT<<")}>> model weight properties in 'bias' " +
      s"feature:\nPREDICTION: ${$(model).getWeights("bias").mkString("\nPREDICTION: ")}")
    var prev = START(0)
    var prev2 = START(1)
    tokenizedSentences.map(sentence => {
      val context: Array[String] = START ++: sentence.tokens.map(normalized) ++: END
      sentence.indexedTokens.zipWithIndex.map { case (IndexedToken(word, begin, end), i) =>
        val tag = $(model).getTagBook.find(_.word == word.toLowerCase).map(_.tag).getOrElse(
          {
            val features = getFeatures(i, word, context, prev, prev2)
            $(model).predict(features)
          }
        )
        prev2 = prev
        prev = tag
        IndexedTaggedWord(word, tag, begin, end)
      }
    }).map(TaggedSentence(_))
  }

  def this() = this(Identifiable.randomUID("POS"))

  def getModel: AveragedPerceptron = $(model)

  def setModel(targetModel: AveragedPerceptron): this.type = set(model, targetModel)

  /** One to one annotation standing from the Tokens perspective, to give each word a corresponding Tag */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val sentences: Array[SentenceToBeTagged] = annotations.collect {
      case sentence: Annotation if sentence.annotatorType == DOCUMENT =>
        /** Creates a sentence bounded by tokens in a sentence */
        val tokenizedSentence = TokenizedSentence(
          annotations.filter(annotation =>
            annotation.annotatorType == TOKEN.toString &&
            annotation.begin >= sentence.begin && annotation.end <= sentence.end
          ).map { a => IndexedToken(a.metadata(TOKEN), a.begin, a.end) }.toArray
        )
        /** Tags the sentence in a token manner while holding sentence bounds. We also hold the original sentence to look for word indexes */
        SentenceToBeTagged(
          tokenizedSentence,
          sentence.begin,
          sentence.end
        )
    }.toArray
    /** Creates an annotation for each word sentence*/
    tag(sentences.map(_.tokenizedSentence))
      .flatMap { case TaggedSentence(_, indexedTaggedWords) =>
        indexedTaggedWords.map { case IndexedTaggedWord(word, tag, begin, end) =>
          Annotation(
            annotatorType,
            begin,
            end,
            Map[String, String]("word" -> word, "tag" -> tag)
          )
        }
      }
  }
}
object PerceptronModel extends DefaultParamsReadable[PerceptronModel]