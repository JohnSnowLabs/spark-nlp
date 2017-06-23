package com.jsl.nlp.annotators.clinical.negex

import com.jsl.nlp.{Annotation, AnnotatorBuilder, Document}
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._

/**
  * Created by Saif Addin on 5/7/2017.
  */
trait NegexTaggerBehaviors { this: FlatSpec =>

  def negexTagger(dataset: => Dataset[Row]): Unit = {
    "A NegexTagger Annotator" should s"successfuly tag " in {
      println(dataset.schema)
      AnnotatorBuilder.withNegexTagger(dataset)
        .collect().foreach {
        row =>
          val document = Document(row.getAs[Row](0))
          println(document)
          row.getSeq[Row](2).map(Annotation(_)).foreach {
            negexAnno =>
              println(negexAnno, negexAnno.metadata.mkString(","))
          }
      }
    }
  }

}
