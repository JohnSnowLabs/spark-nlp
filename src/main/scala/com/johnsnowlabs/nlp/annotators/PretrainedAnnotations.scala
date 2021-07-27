package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserModel
import com.johnsnowlabs.nlp.annotators.parser.typdep.TypedDependencyParserModel
import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel

object PretrainedAnnotations {

  def getPos(annotations: Seq[Annotation], posModelCoordinates: Array[String]): Seq[Annotation] = {
    val pretrainedPosModel = posModelCoordinates.length match {
      case 2 =>
        PerceptronModel.pretrained(name = posModelCoordinates.head, lang = posModelCoordinates(1))
      case 3 =>
        PerceptronModel.pretrained(name = posModelCoordinates.head, lang = posModelCoordinates(1), remoteLoc = posModelCoordinates(2))
      case _ => PerceptronModel.pretrained()
    }
    val averagePerceptron = pretrainedPosModel.model.getOrDefault
    val posTagger = new PerceptronModel().setModel(averagePerceptron)
    posTagger.annotate(annotations)
  }

  def getDependencyParser(annotations: Seq[Annotation], dependencyParserModelCoordinates: Array[String]):
  Seq[Annotation] = {
    val dependencyMaker = dependencyParserModelCoordinates.length match {
      case 2 =>
        DependencyParserModel.pretrained(name = dependencyParserModelCoordinates.head,
          lang = dependencyParserModelCoordinates(1))
      case 3 =>
        DependencyParserModel.pretrained(name = dependencyParserModelCoordinates.head,
          lang = dependencyParserModelCoordinates(1), remoteLoc = dependencyParserModelCoordinates(2))
      case _ => DependencyParserModel.pretrained()
    }
    val dependencyParser = new DependencyParserModel().setPerceptron(dependencyMaker.perceptron.getOrDefault)
    val dependencyParserAnnotations = dependencyParser.annotate(annotations)
    dependencyParserAnnotations
  }

  def getTypedDependencyParser(annotations: Seq[Annotation],
                               typedDependencyParserModelCoordinates: Array[String]): Seq[Annotation] = {
    val pretrainedModel = typedDependencyParserModelCoordinates.length match {
      case 2 =>
        TypedDependencyParserModel.pretrained(name = typedDependencyParserModelCoordinates.head,
          lang = typedDependencyParserModelCoordinates(1))
      case 3 =>
        TypedDependencyParserModel.pretrained(name = typedDependencyParserModelCoordinates.head,
          lang = typedDependencyParserModelCoordinates(1), remoteLoc = typedDependencyParserModelCoordinates(2))
      case _ => TypedDependencyParserModel.pretrained()
    }
    val dependencyPipe = pretrainedModel.trainDependencyPipe.getOrDefault
    val trainOptions = pretrainedModel.trainOptions.getOrDefault
    val typedDependencyParser = new TypedDependencyParserModel()
      .setDependencyPipe(dependencyPipe)
      .setOptions(trainOptions)
      .setConllFormat("2009")
    typedDependencyParser.annotate(annotations)
  }

}