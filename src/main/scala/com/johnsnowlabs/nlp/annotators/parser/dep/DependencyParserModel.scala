package com.johnsnowlabs.nlp.annotators.parser.dep

import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel}
import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.annotators.common.{DependencyParsed, DependencyParsedSentence, PosTagged}
import com.johnsnowlabs.nlp.annotators.common.Annotated.PosTaggedSentence
import com.johnsnowlabs.nlp.annotators.parser.dep.GreedyTransition._
import org.apache.spark.ml.param.StringArrayParam
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

class DependencyParserModel(override val uid: String) extends AnnotatorModel[DependencyParserModel] {
  def this() = this(Identifiable.randomUID(DEPENDENCY))

  override val annotatorType: String = DEPENDENCY

  override val requiredAnnotatorTypes: Array[String] =  Array[String](DOCUMENT, POS, TOKEN)

  val perceptronAsArray: StringArrayParam = new StringArrayParam(this, "perceptronAsArray",
    "List of features for perceptron")

  def setPerceptronAsArray(perceptron: Array[String]): this.type = set(perceptronAsArray, perceptron)

  def getDependencyParsedSentence(sentence: PosTaggedSentence): DependencyParsedSentence = {
    val model = new GreedyTransitionApproach()
    val dependencyParsedSentence = model.parseInPrediction(sentence, $(perceptronAsArray))
    dependencyParsedSentence
  }

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val posTaggedSentences = PosTagged.unpack(annotations)
    val sentencesWithDependency = posTaggedSentences.map{sentence => getDependencyParsedSentence(sentence)}
    val dependencyParser = DependencyParsed.pack(sentencesWithDependency)
    dependencyParser
  }
}

object DependencyParserModel extends DefaultParamsReadable[DependencyParserModel]
