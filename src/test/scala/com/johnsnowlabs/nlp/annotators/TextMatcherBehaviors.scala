package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.{Annotation, AnnotatorBuilder}
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._

trait TextMatcherBehaviors { this: FlatSpec =>

  def fullTextMatcher(dataset: => Dataset[Row]) {
    "An TextMatcher Annotator" should "successfully transform data" in {
      AnnotatorBuilder.withFullTextMatcher(dataset)
        .collect().foreach {
        row =>
          row.getSeq[Row](3)
            .map(Annotation(_))
            .foreach {
              case entity: Annotation if entity.annotatorType == "entity" =>
                println(entity, entity.end)
              case _ => ()
            }
      }
    }
  }
}
