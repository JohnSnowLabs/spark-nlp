package com.johnsnowlabs.nlp.annotators.parser.dep.GreedyTransition

import com.johnsnowlabs.nlp.annotators.common.{DependencyParsedSentence, WordWithDependency}
import com.johnsnowlabs.nlp.annotators.common.Annotated.PosTaggedSentence

object GreedyTransitionApproach {

  def predict(posTagged: PosTaggedSentence, dependencyMaker: DependencyMaker): DependencyParsedSentence = {
    val sentence: Sentence = posTagged.indexedTaggedWords
      .map { item => WordData(item.word, item.tag) }.toList
    val dependencies = dependencyMaker.parse(sentence)
    val words = posTagged.indexedTaggedWords
      .zip(dependencies)
      .map{
        case (word, head) =>
          WordWithDependency(word.word, head, "", word.begin, word.end)
      }

    DependencyParsedSentence(words)
  }

}
