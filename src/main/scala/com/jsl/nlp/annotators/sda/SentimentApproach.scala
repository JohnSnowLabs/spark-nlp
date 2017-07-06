package com.jsl.nlp.annotators.sda

import com.jsl.nlp.annotators.common.TaggedSentence
import com.jsl.nlp.annotators.param.WritableAnnotatorComponent

/**
  * Created by saif on 12/06/2017.
  */

/**
  * Guidelines for a SentimentModel
  * Must be writable
  */
abstract class SentimentApproach extends WritableAnnotatorComponent {

  val description: String

  val requiresLemmas: Boolean

  val requiresPOS: Boolean

  def score(taggedSentences: Array[TaggedSentence]): Double

}
