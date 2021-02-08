package com.johnsnowlabs.nlp.annotators.common

import scala.collection.Map

/**
  * Created by Saif Addin on 5/20/2017.
  */

/** Word tag pair */
case class TaggedWord(word: String, tag: String)

case class IndexedTaggedWord(word: String, tag: String, begin: Int = 0, end: Int = 0,
                             confidence: Option[Float] = None, metadata: Map[String, String] = Map()) {
  def toTaggedWord: TaggedWord = TaggedWord(this.word, this.tag)
}
