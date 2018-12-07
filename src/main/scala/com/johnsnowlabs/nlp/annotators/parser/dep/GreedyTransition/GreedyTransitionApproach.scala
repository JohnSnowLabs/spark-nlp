package com.johnsnowlabs.nlp.annotators.parser.dep.GreedyTransition

import com.johnsnowlabs.nlp.annotators.common.{DependencyParsedSentence, WordWithDependency}
import com.johnsnowlabs.nlp.annotators.common.Annotated.PosTaggedSentence
import com.johnsnowlabs.nlp.annotators.parser.dep.{Perceptron, Tagger}

/**
  * Parser based on the code of Matthew Honnibal and Martin Andrews
  */
class GreedyTransitionApproach {

  def predict(posTagged: PosTaggedSentence, trainedPerceptron: Array[String]): DependencyParsedSentence = {
    val dependencyMaker = loadPerceptronInPrediction(trainedPerceptron)
    val sentence: Sentence = posTagged.indexedTaggedWords
      .map { item => WordData(item.word, item.tag) }.toList
    val dependencies = dependencyMaker.predictHeads(sentence)
    val words = posTagged.indexedTaggedWords
      .zip(dependencies)
      .map{
        case (word, dependency) =>
          WordWithDependency(word.word, word.begin, word.end, dependency)
      }

    DependencyParsedSentence(words)
  }

  def loadPerceptronInPrediction(trainedPerceptron: Array[String]): DependencyMakerPrediction = {
    val dependencyMaker = new DependencyMakerPrediction()
    dependencyMaker.perceptron.load(trainedPerceptron.toIterator)
    dependencyMaker
  }

  def loadPerceptronInTraining(trainedPerceptron: Array[String]): DependencyMakerTraining = {
    val dependencyMaker = new DependencyMakerTraining()
    dependencyMaker.perceptron.load(trainedPerceptron.toIterator)
    dependencyMaker.perceptron.cleanLearning()
    dependencyMaker
  }

}
