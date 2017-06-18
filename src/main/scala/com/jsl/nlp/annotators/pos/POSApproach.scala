package com.jsl.nlp.annotators.pos

import com.jsl.nlp.annotators.common.{TaggedSentence, TokenizedSentence}

/**
  * Created by Saif Addin on 5/13/2017.
  */
abstract class POSApproach {
  val description: String
  val model: POSModel
  def tag(sentences: Array[TokenizedSentence]): Array[TaggedSentence]
}