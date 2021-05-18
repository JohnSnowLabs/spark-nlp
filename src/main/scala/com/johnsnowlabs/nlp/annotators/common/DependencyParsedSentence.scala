package com.johnsnowlabs.nlp.annotators.common

case class DependencyParsedSentence(wordsWithDependency: Array[WordWithDependency])

object DependencyParsedSentence {
  def apply(wordsWithDependency: Array[WordWithDependency]) = new DependencyParsedSentence(wordsWithDependency)
}