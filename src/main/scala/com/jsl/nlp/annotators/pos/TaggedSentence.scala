package com.jsl.nlp.annotators.pos

/**
  * Created by Saif Addin on 5/20/2017.
  */
case class TaggedSentence(taggedWords: List[TaggedWord]) {
  val words: List[String] = taggedWords.map(_.word)
  val tags: List[String] = taggedWords.map(_.tag)
}
