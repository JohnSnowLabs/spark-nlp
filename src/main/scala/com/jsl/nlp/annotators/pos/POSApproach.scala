package com.jsl.nlp.annotators.pos

/**
  * Created by Saif Addin on 5/13/2017.
  */
abstract class POSApproach {

  val description: String

  def tag(tokens: Array[String]): Array[TaggedWord]

}
