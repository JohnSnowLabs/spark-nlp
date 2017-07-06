package com.jsl.nlp.annotators.pos

import com.jsl.nlp.annotators.common.{TaggedSentence, TokenizedSentence}
import com.jsl.nlp.annotators.param.WritableAnnotatorComponent

/**
  * Created by Saif Addin on 5/13/2017.
  */

/**
  * Guideline structure for any approach for POS tagging
  */
abstract class POSApproach extends WritableAnnotatorComponent {
  val description: String
  val model: POSModel[_]
  def tag(sentences: Array[TokenizedSentence]): Array[TaggedSentence]
}