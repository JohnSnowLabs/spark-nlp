package com.jsl.nlp.annotators.pos

/**
  * Created by Saif Addin on 5/20/2017.
  */
case class TaggedSentence(taggedWords: List[TaggedWord]) {
  def words = taggedWords.map(_.word)
  def tags = taggedWords.map(_.tag)
}
