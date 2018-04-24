package com.johnsnowlabs.nlp.annotators.datasets

import com.johnsnowlabs.nlp.Annotation
import org.apache.spark.sql.Row

/**
  * Created by jose on 12/01/18.
  */
case class AssertionAnnotationAndText(text: String, target: String, label: String, start:Int, end:Int)
case class AssertionAnnotationWithLabel(label: String, start:Int, end:Int)
object AssertionAnnotationWithLabel {
  def fromNer(label: String, ner: Seq[Row]): Seq[AssertionAnnotationWithLabel] = {
    ner.map{n => {
      val annotation = Annotation(n)
      AssertionAnnotationWithLabel(label, annotation.begin, annotation.end)
    }}
  }
}