package com.johnsnowlabs.nlp.annotators.datasets

import com.johnsnowlabs.nlp.Annotation

/**
  * Created by jose on 12/01/18.
  */
case class AssertionAnnotationAndText(text: String, target: String, label: String, start:Int, end:Int)
case class AssertionAnnotationWithLabel(label: String, start:Int, end:Int)
object AssertionAnnotationWithLabel {
  def fromNer(label: String, ner: Seq[Annotation]): Seq[AssertionAnnotationWithLabel] = {
    ner.map{n => {
      AssertionAnnotationWithLabel(label, n.begin, n.end)
    }}
  }
}