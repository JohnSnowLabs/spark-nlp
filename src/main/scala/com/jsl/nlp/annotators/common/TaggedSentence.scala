package com.jsl.nlp.annotators.common

/**
  * Created by Saif Addin on 5/20/2017.
  */

/**
  * Structure to hold Sentences as list of words and POS-tags
  * @param taggedWords Word tag pairs
  */
case class TaggedSentence(taggedWords: Array[TaggedWord]) {
  /** Recurrently needed to access all words */
  val words: Array[String] = taggedWords.map(_.word)
  /** Recurrently needed to access all tags */
  val tags: Array[String] = taggedWords.map(_.tag)
  /** ready function to return pairwise tagged words */
  def mapWords: Map[String, String] = words.zip(tags).toMap
}