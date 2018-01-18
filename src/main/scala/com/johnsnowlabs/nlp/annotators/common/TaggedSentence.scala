package com.johnsnowlabs.nlp.annotators.common

/**
  * Created by Saif Addin on 5/20/2017.
  */

/**
  * Structure to hold Sentences as list of words and POS-tags
  * @param taggedWords Word tag pairs
  */
case class TaggedSentence(taggedWords: Array[TaggedWord], indexedTaggedWords: Array[IndexedTaggedWord] = Array()) {
  def this(indexedTaggedWords: Array[IndexedTaggedWord]) = this(indexedTaggedWords.map(_.toTaggedWord), indexedTaggedWords)
  /** Recurrently needed to access all words */
  val words: Array[String] = taggedWords.map(_.word)
  /** Recurrently needed to access all tags */
  val tags: Array[String] = taggedWords.map(_.tag)
  /** ready function to return pairwise tagged words */
  def tupleWords: Array[(String, String)] = words.zip(tags)
  def mapWords: Map[String, String] = tupleWords.toMap
}

object TaggedSentence {
  def apply(indexedTaggedWords: Array[IndexedTaggedWord]) = new TaggedSentence(indexedTaggedWords.map(_.toTaggedWord), indexedTaggedWords)
}