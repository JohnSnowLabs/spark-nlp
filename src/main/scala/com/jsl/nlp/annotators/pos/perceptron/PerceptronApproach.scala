package com.jsl.nlp.annotators.pos.perceptron

import com.jsl.nlp.annotators.pos.{POSApproach, TaggedWord}

/**
  * Created by Saif Addin on 5/17/2017.
  * Inspired on Averaged Perceptron by Matthew Honnibal
  * https://explosion.ai/blog/part-of-speech-pos-tagger-in-python
  */
class PerceptronApproach extends POSApproach {

  override val description: String = "Averaged Perceptron tagger, iterative average weights upon training"

  override def tag(tokens: Array[String]): Array[TaggedWord] = {
    ???
  }

}
