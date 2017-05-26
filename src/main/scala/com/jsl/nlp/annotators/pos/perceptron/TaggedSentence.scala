package com.jsl.nlp.annotators.pos.perceptron

/**
  * Created by Saif Addin on 5/20/2017.
  */
case class TaggedSentence(words: List[String], tags: List[String]) {
  def tagged: List[TaggedWord] = words.zip(tags).map(t => TaggedWord(t._1, t._2))
}
