package com.jsl.nlp.annotators.common

/**
  * Created by Saif Addin on 6/18/2017.
  */

/** Internal structure for a sentence that is split into tokens */
case class TokenizedSentence(tokens: Array[String]) {
  def condense = tokens.mkString(" ")
}
