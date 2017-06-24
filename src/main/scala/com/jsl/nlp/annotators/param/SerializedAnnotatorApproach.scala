package com.jsl.nlp.annotators.param

/**
  * Created by saif on 24/06/17.
  */
trait SerializedAnnotatorApproach[TargetAnnotatorApproach] {
  val id: String
  def deserialize: TargetAnnotatorApproach
}
