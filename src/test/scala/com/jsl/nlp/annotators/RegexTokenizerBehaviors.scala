package com.jsl.nlp.annotators

import com.jsl.nlp.{Annotation, AnnotatorBuilder, Document}
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._

trait RegexTokenizerBehaviors { this: FlatSpec =>

  def fullTokenizerPipeline(dataset: => Dataset[Row]) {
    "A RegexTokenizer Annotator" should "successfully transform data" in {
      println(dataset.schema)
      val tokenizedData = AnnotatorBuilder.withTokenizer(dataset)
      tokenizedData
        .collect().foreach {
        row =>
          val document = Document(row.getAs[Row](0))
          println(document)
          row.getSeq[Row](1).map(Annotation(_)).foreach {
            token =>
              println(token, document.text.substring(token.begin, token.end))
          }
      }
    }
  }

  /**
    * Add more tokenizer behavior tests here ...
    */

}
