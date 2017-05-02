package com.jsl.nlp.annotators

import com.jsl.nlp.{Annotation, AnnotatorBuilder, Document, SparkBasedTest}
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._

trait EntityExtractorBehaviors extends SparkBasedTest { this: FlatSpec =>

  def fullEntityExtractorPipeline(dataset: => Dataset[Row]) {
    "An EntityExtractor Annotator" should "successfully transform data" in {
      println(dataset.schema)
      AnnotatorBuilder.withFullEntityExtractor(dataset)
        .collect().foreach {
        row =>
          val document = Document(row.getAs[Row](0))
          println(document)
          row.getSeq[Row](4)
            .map(Annotation(_))
            .foreach {
              case entity: Annotation if entity.aType == "entity" =>
                println(entity, document.text.substring(entity.begin, entity.end))
              case _ => ()
            }
      }
    }
  }
}
