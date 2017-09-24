package com.jsl.nlp.annotators.ner

import com.jsl.nlp.{AnnotatorBuilder, AnnotatorType}
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._
import Matchers._


trait NERTaggerBehaviors { this: FlatSpec =>

  def fullNERTaggerPipeline(dataset: => Dataset[Row]) = {
    "NER Tagger Annotator" should "successfully annotate" in {
      val df = AnnotatorBuilder.withNERTagger(dataset)
      df.select("document", "ner").show(10)
      val annotations = df.select("document", "ner").foreach( ds => {
        val annotations = ds.getAs[Seq[Row]](1)
        annotations.size should be > 0
        annotations.foreach( rowAnnotation => {
          val annotatorType = rowAnnotation.getAs[String](0)
          annotatorType should be (AnnotatorType.NAMED_ENTITY)
        })
      })
    }
  }
}
