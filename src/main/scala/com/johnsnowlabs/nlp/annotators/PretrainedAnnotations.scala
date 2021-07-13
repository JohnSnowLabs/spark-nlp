package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserModel
import com.johnsnowlabs.nlp.annotators.parser.typdep.TypedDependencyParserModel
import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel

object PretrainedAnnotations {

  def getPos(annotations: Seq[Annotation], posModel: Option[PerceptronModel]): Seq[Annotation] = {
    val pretrainedPosModel = PerceptronModel.pretrained()
    val averagePerceptron = posModel.getOrElse(pretrainedPosModel).model.getOrDefault
    val posTagger = new PerceptronModel().setModel(averagePerceptron)
    posTagger.annotate(annotations)
  }

  def getDependencyParser(annotations: Seq[Annotation], dependencyParserModel: Option[DependencyParserModel]):
  Seq[Annotation] = {
    val dependencyMaker = dependencyParserModel.getOrElse(DependencyParserModel.pretrained()).perceptron.getOrDefault
    val dependencyParser = new DependencyParserModel().setPerceptron(dependencyMaker)
    val dependencyParserAnnotations = dependencyParser.annotate(annotations)
    dependencyParserAnnotations
  }

  def getTypedDependencyParser(annotations: Seq[Annotation],
                               typedDependencyParserModel: Option[TypedDependencyParserModel]): Seq[Annotation] = {
    val pretrainedModel = typedDependencyParserModel.getOrElse(TypedDependencyParserModel.pretrained())
    val dependencyPipe = pretrainedModel.trainDependencyPipe.getOrDefault
    val trainOptions = pretrainedModel.trainOptions.getOrDefault
    val typedDependencyParser = new TypedDependencyParserModel()
      .setDependencyPipe(dependencyPipe)
      .setOptions(trainOptions)
      .setConllFormat("2009")
    typedDependencyParser.annotate(annotations)
  }

}