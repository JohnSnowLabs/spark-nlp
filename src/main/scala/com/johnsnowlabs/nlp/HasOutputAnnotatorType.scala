package com.johnsnowlabs.nlp

trait HasOutputAnnotatorType {
  type AnnotatorType = String
  val outputAnnotatorType: AnnotatorType
}
