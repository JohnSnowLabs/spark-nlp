package com.jsl.nlp.annotators.common

import com.jsl.nlp.annotators.param.SerializedAnnotatorApproach

/**
  * Created by saif on 24/06/17.
  */
trait AnnotatorApproach {
  def serialize: SerializedAnnotatorApproach[_ <: AnnotatorApproach]
}