package com.jsl.nlp.annotators

import java.util.Calendar

import com.jsl.nlp.{Annotation, AnnotatorBuilder, Document}
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._

/**
  * Created by Saif Addin on 6/6/2017.
  */
trait DateMatcherBehaviors extends FlatSpec {

  def sparkBasedDateMatcher(dataset: => Dataset[Row]): Unit = {
    "A DateMatcher Annotator" should s"successfuly parse dates}" in {
      println(dataset.schema)
      AnnotatorBuilder.withDateMatcher(dataset)
        .collect().foreach {
        row =>
          val document = Document(row.getAs[Row](0))
          println(document)
          row.getSeq[Row](1).map(Annotation(_)).foreach {
            matchedAnnotation =>
              println(matchedAnnotation, matchedAnnotation.metadata.mkString(","))
          }
      }
    }
  }

}
