package com.jsl.nlp.annotators.pos.perceptron

/**
  * Created by Saif Addin on 5/20/2017.
  */
case class TaggedSentence(words: List[Word], tags: List[String]) {
  def tagged: List[TaggedWord] = words.zip(tags).map(t => TaggedWord(t._1.word, t._2))
}
