package com.johnsnowlabs.nlp.annotators.param

/**
  * Created by saif on 24/06/17.
  */
trait WritableAnnotatorComponent extends Serializable {
  def serialize: SerializedAnnotatorComponent[_ <: WritableAnnotatorComponent]
}