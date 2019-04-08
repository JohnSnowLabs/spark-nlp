package com.johnsnowlabs.nlp.annotators.common

/**
  * Created by Saif Addin on 5/20/2017.
  */

/** Word tag pair */
case class TaggedWord(word: String, tag: String)

case class IndexedTaggedWord(word: String, tag: String, begin: Int = 0, end: Int = 0, confidence: Option[Float] = None) {
  def toTaggedWord: TaggedWord = TaggedWord(this.word, this.tag)
}
