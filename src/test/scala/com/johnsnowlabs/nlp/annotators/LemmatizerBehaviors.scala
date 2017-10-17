package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.{Annotation, AnnotatorBuilder, AnnotatorType}
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._

/**
  * Created by saif on 01/05/17.
  */
trait LemmatizerBehaviors { this: FlatSpec =>

  def fullLemmatizerPipeline(dataset: => Dataset[Row]) {
    "a Lemmatizer Annotator" should "succesfully transform data" in {
      dataset.show
      AnnotatorBuilder.withFullLemmatizer(dataset)
        .collect().foreach {
        row =>
          row.getSeq[Row](2)
            .map(Annotation(_))
            .foreach {
              case lemma: Annotation if lemma.annotatorType == AnnotatorType.TOKEN =>
                println(lemma, lemma.result)
              case _ => ()
            }
      }
    }
  }
}
