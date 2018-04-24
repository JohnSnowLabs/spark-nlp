package com.johnsnowlabs.nlp.annotators.datasets

import com.johnsnowlabs.nlp.Annotation
import org.apache.spark.sql.Row

/**
  * Created by jose on 12/01/18.
  */
case class AssertionAnnotationAndText(text: String, target: String, label: String, start:Int, end:Int)
case class AssertionAnnotationWithLabel(label: String, start:Int, end:Int)
object AssertionAnnotationWithLabel {
  def fromNer(doc: String, label: String, ner: Seq[Row]): Seq[AssertionAnnotationWithLabel] = {
    ner.map{n => {
      val annotation = Annotation(n)
      val content = doc.split(" ")
      val target = doc.substring(annotation.begin, annotation.end).split(" ")
      val start = content.indexOf(target.head)
      val end = content.indexOf(target.last)
      AssertionAnnotationWithLabel(label, start, end)
    }}
  }
}