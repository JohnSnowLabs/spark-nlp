package com.johnsnowlabs.nlp.annotators.param

/**
  * Created by saif on 24/06/17.
  */
trait SerializedAnnotatorComponent[T <: WritableAnnotatorComponent] {
  def deserialize: T
}
