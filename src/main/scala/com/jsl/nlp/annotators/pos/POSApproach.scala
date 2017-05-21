package com.jsl.nlp.annotators.pos

import com.jsl.nlp.annotators.pos.perceptron.TaggedWord

/**
  * Created by Saif Addin on 5/13/2017.
  */
abstract class POSApproach {
  val description: String
  def tag(sentences: Array[String]): Array[TaggedWord]
}
object POSApproach {
  var model: Option[POSModel] = None
  def isTrained: Boolean = model.isDefined
}
