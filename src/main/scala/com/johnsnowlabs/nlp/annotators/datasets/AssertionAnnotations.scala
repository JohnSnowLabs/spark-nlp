package com.johnsnowlabs.nlp.annotators.datasets

import com.johnsnowlabs.nlp.Annotation
import org.apache.spark.sql.Row

import scala.collection.mutable.ArrayBuffer

/**
  * Created by jose on 12/01/18.
  */
case class AssertionAnnotationAndText(text: String, target: String, label: String, start:Int, end:Int)
case class AssertionAnnotationWithLabel(label: String, start:Int, end:Int)
object AssertionAnnotationWithLabel {
  def fromNer(doc: String, label: String, ner: Seq[Row], targetLabels: Array[String]): Seq[AssertionAnnotationWithLabel] = {
    val annotations = ner.map { r => Annotation(r) }
    val targets = annotations.zipWithIndex.filter(a => targetLabels.contains(a._1.result)).toIterator
    val ranges = ArrayBuffer.empty[(Int, Int)]
    while (targets.hasNext) {
      val annotation = targets.next
      var range = (annotation._1.begin, annotation._1.end)
      var look = true
      while(look && targets.hasNext) {
        val nextAnnotation = targets.next
        if (nextAnnotation._2 == annotation._2 + 1)
          range = (range._1, nextAnnotation._1.end)
        else
          look = false
      }
      ranges.append(range)
    }
    if (ranges.nonEmpty) {
      ranges.map { range => {
        val content = doc.split(" ")
        val target = doc.substring(range._1, range._2+1).split(" ")
        val start = content.indexOf(target.head)
        val end = content.indexOf(target.last)
        AssertionAnnotationWithLabel(label, start, end)
      }}
    }
    else
      throw new IllegalArgumentException("NER Based assertion status failed due to missing entities in nerCol")
  }
}