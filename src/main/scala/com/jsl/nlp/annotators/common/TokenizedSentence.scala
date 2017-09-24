package com.jsl.nlp.annotators.common

/**
  * Created by Saif Addin on 6/18/2017.
  */

/** Internal structure for a sentence that is split into tokens */
case class TokenizedSentence(tokens: Array[String], indexedTokens: Array[IndexedToken]) {
  def this(tokens: Array[String]) = this(tokens, tokens.map(IndexedToken(_)))
  def this(indexedTokens: Array[IndexedToken]) = this(indexedTokens.map(_.token), indexedTokens)

  def condense = tokens.mkString(" ")
}

object TokenizedSentence {
  def apply(tokens: Array[String]) = new TokenizedSentence(tokens, tokens.map(IndexedToken(_)))
  def apply(indexedTokens: Array[IndexedToken]) = new TokenizedSentence(indexedTokens.map(_.token), indexedTokens)
}
