package com.jsl.nlp.annotators

import com.jsl.nlp.{Annotation, AnnotatorBuilder, Document}
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._

trait StemmerBehaviors { this: FlatSpec =>

  def fullStemmerPipeline(dataset: => Dataset[Row]) {
    "A Stemmer Annotator" should "successfully transform data" in {
      println(dataset.schema)
      AnnotatorBuilder.withFullStemmer(dataset)
        .collect.foreach {
        row =>
          val document = Document(row.getAs[Row](0))
          println(document)
          row.getSeq[Row](2)
            .map(Annotation(_))
            .foreach {
              case stem: Annotation if stem.aType == Stemmer.aType =>
                println(stem, document.text.substring(stem.begin, stem.end))
              case _ => ()
            }
      }
    }
  }
}
