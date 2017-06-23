package com.jsl.nlp.annotators.ner

import com.jsl.nlp.{Annotation, AnnotatorBuilder, Document}
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._

trait NERTaggerBehaviors { this: FlatSpec =>

  def fullNERTaggerPipeline(dataset: => Dataset[Row]) {
    "A NERTagger Annotator" should "successfully transform data" in {
      println(dataset.schema)
      AnnotatorBuilder.withNERTagger(dataset)
        .collect.foreach {
        row =>
          val document = Document(row.getAs[Row](0))
          println(document)
          row.getSeq[Row](3)
            .map(Annotation(_))
            .foreach {
              case nertag: Annotation if nertag.aType == NERTagger.aType =>
                println(nertag, nertag.metadata(NERTagger.aType), document.text.substring(nertag.begin, nertag.end))
              case _ => ()
            }
      }
    }
  }
}
