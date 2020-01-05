package com.johnsnowlabs.nlp.annotators.btm

import com.johnsnowlabs.nlp.{Annotation, AnnotatorBuilder}
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._

trait BigTextMatcherBehaviors { this: FlatSpec =>

  def fullBigTextMatcher(dataset: => Dataset[Row]) {
    "An BigTextMatcher Annotator" should "successfully transform data" in {
      AnnotatorBuilder.withFullBigTextMatcher(dataset)
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
