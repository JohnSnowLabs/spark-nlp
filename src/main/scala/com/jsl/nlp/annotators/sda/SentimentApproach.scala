package com.jsl.nlp.annotators.sda

import com.jsl.nlp.annotators.common.{WritableAnnotatorComponent, TaggedSentence}

/**
  * Created by saif1_000 on 12/06/2017.
  */
abstract class SentimentApproach extends WritableAnnotatorComponent {

  val description: String

  val requiresLemmas: Boolean

  val requiresPOS: Boolean

  def score(taggedSentences: Array[TaggedSentence]): Double

}
