package com.johnsnowlabs.nlp.annotators.common

/**
  * Created by Saif Addin on 6/18/2017.
  */

/** Internal structure for a sentence that is split into tokens */
case class TokenizedSentence(indexedTokens: Array[IndexedToken], sentenceIndex: Int) {
  lazy val tokens = indexedTokens.map(t => t.token)

  def condense = tokens.mkString(" ")
}