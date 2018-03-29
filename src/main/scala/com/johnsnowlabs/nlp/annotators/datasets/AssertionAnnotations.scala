package com.johnsnowlabs.nlp.annotators.datasets

/**
  * Created by jose on 12/01/18.
  */
case class AssertionAnnotationAndText(text: String, target: String, label: String, start:Int, end:Int)
case class AssertionAnnotationWithLabel(label: String, start:Int, end:Int)