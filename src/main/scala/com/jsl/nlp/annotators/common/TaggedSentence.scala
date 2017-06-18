package com.jsl.nlp.annotators.common

/**
  * Created by Saif Addin on 5/20/2017.
  */
case class TaggedSentence(taggedWords: Array[TaggedWord]) {
  val words: Array[String] = taggedWords.map(_.word)
  val tags: Array[String] = taggedWords.map(_.tag)
  def mapWords: Map[String, String] = words.zip(tags).toMap
}
