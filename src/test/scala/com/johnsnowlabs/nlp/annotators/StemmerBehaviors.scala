package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.{Annotation, AnnotatorBuilder, AnnotatorType}
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._

trait StemmerBehaviors { this: FlatSpec =>

  def fullStemmerPipeline(dataset: => Dataset[Row]) {
    "A Stemmer Annotator" should "successfully transform data" taggedAs FastTest in {
      AnnotatorBuilder.withFullStemmer(dataset)
        .collect.foreach {
        row =>
          row.getSeq[Row](2)
            .map(Annotation(_))
            .foreach {
              case stem: Annotation if stem.annotatorType == AnnotatorType.TOKEN =>
                println(stem, stem.result)
              case _ => ()
            }
      }
    }
  }
}
