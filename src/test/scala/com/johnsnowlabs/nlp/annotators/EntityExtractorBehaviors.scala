package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.{Annotation, AnnotatorBuilder}
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._

trait EntityExtractorBehaviors { this: FlatSpec =>

  def fullEntityExtractor(dataset: => Dataset[Row]) {
    "An EntityExtractor Annotator" should "successfully transform data" in {
      AnnotatorBuilder.withFullEntityExtractor(dataset)
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
