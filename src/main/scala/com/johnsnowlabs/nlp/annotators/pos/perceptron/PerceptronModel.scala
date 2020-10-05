package com.johnsnowlabs.nlp.annotators.pos.perceptron

import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.serialization.StructFeature
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, HasPretrained, ParamsAndFeaturesReadable, WithAnnotate}
import org.apache.spark.ml.util.Identifiable

/**
  * Part of speech tagger that might use different approaches
  *
  * See [[https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/nlp/annotators/pos/perceptron]] for further reference on how to use this API.
  *
  * @param uid Internal constructor requirement for serialization of params
  * @@model: representation of a POS Tagger approach
  * @groupname anno Annotator types
  * @groupdesc anno Required input and expected output annotator types
  * @groupname Ungrouped Members
  * @groupname param Parameters
  * @groupname setParam Parameter setters
  * @groupname getParam Parameter getters
  * @groupname Ungrouped Members
  * @groupprio param  1
  * @groupprio anno  2
  * @groupprio Ungrouped 3
  * @groupprio setParam  4
  * @groupprio getParam  5
  * @groupdesc Parameters A list of (hyper-)parameter keys this annotator can take. Users can set and get the parameter values through setters and getters, respectively.
  */
class PerceptronModel(override val uid: String) extends AnnotatorModel[PerceptronModel] with WithAnnotate[PerceptronModel] with PerceptronUtils {

  import com.johnsnowlabs.nlp.AnnotatorType._

  /** Internal structure for target sentences holding their range information which is used for annotation */
  private case class SentenceToBeTagged(tokenizedSentence: TokenizedSentence, start: Int, end: Int)


  /** POS model
    *
    * @group param
    **/
  val model: StructFeature[AveragedPerceptron] = new StructFeature[AveragedPerceptron](this, "POS Model")
  /** Output annotator types : POS
    *
    * @group anno
    **/
  override val outputAnnotatorType: AnnotatorType = POS
  /** Input annotator types : TOKEN, DOCUMENT
    *
    * @group anno
    **/
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN, DOCUMENT)

  /**
    * Tags a group of sentences into POS tagged sentences
    * The logic here is to create a sentence context, run through every word and evaluate its context
    * Based on how frequent a context appears around a word, such context is given a score which is used to predict
    * Some words are marked as non ambiguous from the beginning
    *
    * @param tokenizedSentences Sentence in the form of single word tokens
    * @return A list of sentences which have every word tagged
    */
  def tag(tokenizedSentences: Array[TokenizedSentence]): Array[TaggedSentence] = {
    //logger.debug(s"PREDICTION: Tagging:\nSENT: <<${tokenizedSentences.map(_.condense).mkString(">>\nSENT<<")}>> model weight properties in 'bias' " +
      //s"feature:\nPREDICTION: ${$$(model).getWeights("bias").mkString("\nPREDICTION: ")}")
    var prev = START(0)
    var prev2 = START(1)
    tokenizedSentences.map(sentence => {
      val context: Array[String] = START ++: sentence.tokens.map(normalized) ++: END
      sentence.indexedTokens.zipWithIndex.map { case (IndexedToken(word, begin, end), i) =>
        val tag = $$(model).getTaggedBook.getOrElse(word.toLowerCase,
          {
            val features = getFeatures(i, word, context, prev, prev2)
            $$(model).predict(features)
          }
        )
        prev2 = prev
        prev = tag
        IndexedTaggedWord(word, tag, begin, end)
      }
    }).map(TaggedSentence(_))
  }

  def this() = this(Identifiable.randomUID("POS"))

  /** @group getParam */
  def getModel: AveragedPerceptron = $$(model)

  /** @group setParam */
  def setModel(targetModel: AveragedPerceptron): this.type = set(model, targetModel)

  /** One to one annotation standing from the Tokens perspective, to give each word a corresponding Tag */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val tokenizedSentences = TokenizedWithSentence.unpack(annotations)
    val tagged = tag(tokenizedSentences.toArray)
    PosTagged.pack(tagged)
  }
}

trait ReadablePretrainedPerceptron extends ParamsAndFeaturesReadable[PerceptronModel] with HasPretrained[PerceptronModel] {
  override val defaultModelName = Some("pos_anc")
  /** Java compliant-overrides */
  override def pretrained(): PerceptronModel = super.pretrained()
  override def pretrained(name: String): PerceptronModel = super.pretrained(name)
  override def pretrained(name: String, lang: String): PerceptronModel = super.pretrained(name, lang)
  override def pretrained(name: String, lang: String, remoteLoc: String): PerceptronModel = super.pretrained(name, lang, remoteLoc)
}

object PerceptronModel extends ReadablePretrainedPerceptron