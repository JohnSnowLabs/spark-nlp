package com.jsl.nlp.annotators.pos

import com.jsl.nlp.annotators.common.{AnnotatorApproach, TaggedSentence, TokenizedSentence}

/**
  * Created by Saif Addin on 5/13/2017.
  */
abstract class POSApproach extends AnnotatorApproach {
  val description: String
  val model: POSModel
  def tag(sentences: Array[TokenizedSentence]): Array[TaggedSentence]
}