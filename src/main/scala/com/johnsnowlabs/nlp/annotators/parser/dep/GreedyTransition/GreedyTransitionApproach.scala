package com.johnsnowlabs.nlp.annotators.parser.dep.GreedyTransition

import com.johnsnowlabs.nlp.annotators.common.{DependencyParsedSentence, WordWithDependency}
import com.johnsnowlabs.nlp.annotators.common.Annotated.PosTaggedSentence
import com.johnsnowlabs.nlp.annotators.parser.dep.Tagger

/**
  * Parser based on the code of Matthew Honnibal and Martin Andrews
  */
class GreedyTransitionApproach {

  def predict(posTagged: PosTaggedSentence, trainedPerceptron: Array[String], tagger: Tagger): DependencyParsedSentence = {
    val sentence: Sentence = posTagged.indexedTaggedWords
      .map { item => WordData(item.word, item.tag) }.toList
    val dependencyMaker = loadPerceptronInPrediction(trainedPerceptron, tagger)
    val dependencies = dependencyMaker.process(sentence, false)
    val words = posTagged.indexedTaggedWords
      .zip(dependencies)
      .map{
        case (word, dependency) =>
          WordWithDependency(word.word, word.begin, word.end, dependency)
      }

    DependencyParsedSentence(words)
  }

  def loadPerceptronInPrediction(trainedPerceptron: Array[String], tagger: Tagger): DependencyMaker = {
    val dependencyMaker = new DependencyMaker(tagger)
    dependencyMaker.perceptron.load(trainedPerceptron.toIterator)
    dependencyMaker
  }

}
