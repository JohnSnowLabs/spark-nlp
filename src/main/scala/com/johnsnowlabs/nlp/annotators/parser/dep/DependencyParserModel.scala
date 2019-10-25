package com.johnsnowlabs.nlp.annotators.parser.dep

import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, HasPretrained, ParamsAndFeaturesReadable}
import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.annotators.common.{DependencyParsed, DependencyParsedSentence, PosTagged}
import com.johnsnowlabs.nlp.annotators.common.Annotated.PosTaggedSentence
import com.johnsnowlabs.nlp.annotators.parser.dep.GreedyTransition._
import com.johnsnowlabs.nlp.pretrained.ResourceDownloader
import com.johnsnowlabs.nlp.serialization.StructFeature
import org.apache.spark.ml.util.Identifiable

class DependencyParserModel(override val uid: String) extends AnnotatorModel[DependencyParserModel] {
  def this() = this(Identifiable.randomUID(DEPENDENCY))

  override val outputAnnotatorType: String = DEPENDENCY

  override val inputAnnotatorTypes: Array[String] =  Array[String](DOCUMENT, POS, TOKEN)

  val perceptron: StructFeature[DependencyMaker] = new StructFeature[DependencyMaker](this, "perceptron")

  def setPerceptron(value: DependencyMaker): this.type = set(perceptron, value)

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
  override protected val defaultModelName: String = "dependency_conllu"
}

object DependencyParserModel extends ReadablePretrainedDependency
