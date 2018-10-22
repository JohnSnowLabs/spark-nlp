package com.johnsnowlabs.nlp.annotators.parser.dep

import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel}
import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.annotators.common.{DependencyParsed, DependencyParsedSentence, PosTagged}
import com.johnsnowlabs.nlp.annotators.common.Annotated.PosTaggedSentence
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.annotators.parser.dep.GreedyTransition._
import com.johnsnowlabs.nlp.serialization.ArrayFeature
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

class DependencyParserModel(override val uid: String) extends AnnotatorModel[DependencyParserModel] {
  def this() = this(Identifiable.randomUID(DEPENDENCY))

  override val annotatorType: String = DEPENDENCY

  override val requiredAnnotatorTypes: Array[String] =  Array[String](DOCUMENT, POS, TOKEN)

  val source = new ExternalResourceParam(this, "source", "source file for dependency model")

  def setSourcePath(value: ExternalResource): this.type = set(source, value)

  val perceptronAsArray: ArrayFeature[String] = new ArrayFeature[String](this, "perceptronAsArray")

  def setPerceptronAsArray(perceptron: Array[String]): this.type = set(perceptronAsArray, perceptron)

  def getDependencyParsedSentence(sentence: PosTaggedSentence): DependencyParsedSentence = {
    val model = new GreedyTransitionApproach()
    // val dependencyParsedSentence = model.parseOld(sentence, $(source))
    val dependencyParsedSentence = model.parse(sentence, $$(perceptronAsArray))
    dependencyParsedSentence
  }

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val posTaggedSentences = PosTagged.unpack(annotations)
    val sentencesWithDependency = posTaggedSentences.map{sentence => getDependencyParsedSentence(sentence)}
    DependencyParsed.pack(sentencesWithDependency)
  }
}

object DependencyParserModel extends DefaultParamsReadable[DependencyParserModel]
