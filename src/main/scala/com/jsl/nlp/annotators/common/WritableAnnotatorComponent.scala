package com.jsl.nlp.annotators.common

import com.jsl.nlp.annotators.param.SerializedAnnotatorComponent

/**
  * Created by saif on 24/06/17.
  */
trait WritableAnnotatorComponent {
  def serialize: SerializedAnnotatorComponent[_ <: WritableAnnotatorComponent]
}