package com.johnsnowlabs.nlp

trait HasAnnotatorType {
  type AnnotatorType = String
  val annotatorType: AnnotatorType
}
