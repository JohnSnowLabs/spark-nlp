package com.jsl.nlp.annotators

import com.jsl.nlp.{Annotation, AnnotatorBuilder, Document}
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._

trait NormalizerBehaviors { this: FlatSpec =>

  def fullNormalizerPipeline(dataset: => Dataset[Row]) {
    "A Normalizer Annotator" should "successfully transform data" in {
    println(dataset.schema)
    AnnotatorBuilder.withFullNormalizer(dataset)
      .collect().foreach {
      row =>
        val document = Document(row.getAs[Row](0))
        println(document)
        row.getSeq[Row](3)
          .map(Annotation(_))
          .foreach {
            case stem: Annotation if stem.annotatorType == Normalizer.annotatorType =>
              println(stem, document.text.substring(stem.begin, stem.end), stem.metadata.mkString(", "))
            case _ => ()
          }
      }
    }
  }
}
