package com.jsl.nlp

trait HasAnnotatorType {
  type AnnotatorType = String
  val annotatorType: AnnotatorType
}
