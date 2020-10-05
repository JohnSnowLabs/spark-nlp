package com.johnsnowlabs.nlp.annotators.parser.dep

import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.annotators.common.Annotated.PosTaggedSentence
import com.johnsnowlabs.nlp.annotators.common.{DependencyParsed, DependencyParsedSentence, PosTagged}
import com.johnsnowlabs.nlp.annotators.parser.dep.GreedyTransition._
import com.johnsnowlabs.nlp.serialization.StructFeature
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, HasPretrained, ParamsAndFeaturesReadable, HasSimpleAnnotate}
import org.apache.spark.ml.util.Identifiable

/** Unlabeled parser that finds a grammatical relation between two words in a sentence. Its input is a directory with dependency treebank files.
  *
  * See [[https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/nlp/annotators/parser/dep]] for further reference on how to use this API.
  *
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
  **/
class DependencyParserModel(override val uid: String) extends AnnotatorModel[DependencyParserModel] with HasSimpleAnnotate[DependencyParserModel] {
  def this() = this(Identifiable.randomUID(DEPENDENCY))

  /** Output annotation type : DEPENDENCY
    *
    * @group anno
    **/
  override val outputAnnotatorType: String = DEPENDENCY

  /** Input annotation type : DOCUMENT, POS, TOKEN
    *
    * @group anno
    **/
  override val inputAnnotatorTypes: Array[String] = Array[String](DOCUMENT, POS, TOKEN)


  /** @group param */
  val perceptron: StructFeature[DependencyMaker] = new StructFeature[DependencyMaker](this, "perceptron")

  /** @group setParam */
  def setPerceptron(value: DependencyMaker): this.type = set(perceptron, value)

  /** @group getParam */
  def getDependencyParsedSentence(sentence: PosTaggedSentence): DependencyParsedSentence = {
    val dependencyParsedSentence = GreedyTransitionApproach.predict(sentence, $$(perceptron))
    dependencyParsedSentence
  }

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val posTaggedSentences = PosTagged.unpack(annotations)
    val sentencesWithDependency = posTaggedSentences.map{sentence => getDependencyParsedSentence(sentence)}
    val dependencyParser = DependencyParsed.pack(sentencesWithDependency)
    dependencyParser
  }
}

trait ReadablePretrainedDependency extends ParamsAndFeaturesReadable[DependencyParserModel] with HasPretrained[DependencyParserModel] {
  override val defaultModelName = Some("dependency_conllu")
  /** Java compliant-overrides */
  override def pretrained(): DependencyParserModel = super.pretrained()
  override def pretrained(name: String): DependencyParserModel = super.pretrained(name)
  override def pretrained(name: String, lang: String): DependencyParserModel = super.pretrained(name, lang)
  override def pretrained(name: String, lang: String, remoteLoc: String): DependencyParserModel = super.pretrained(name, lang, remoteLoc)
}

object DependencyParserModel extends ReadablePretrainedDependency
