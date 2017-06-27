package com.jsl.nlp.annotators.sda.pragmatic

import com.jsl.nlp.annotators.common.TaggedSentence
import com.jsl.nlp.{Annotation, AnnotatorBuilder, Document}
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._

/**
  * Created by Saif Addin on 5/7/2017.
  */
trait PragmaticSentimentBehaviors { this: FlatSpec =>

  def isolatedSentimentDetector(taggedSentences: Array[TaggedSentence], expectedScore: Double): Unit = {
    s"tagged sentences" should s"have an expected score of $expectedScore" in {
      val pragmaticScorer = new PragmaticScorer
      val result = pragmaticScorer.score(taggedSentences)
      assert(result == expectedScore, s"because result: $result did not match expected: $expectedScore")
    }
  }

  def sparkBasedSentimentDetector(dataset: => Dataset[Row]): Unit = {
    "a Pragmatic Sentiment Analysis Annotator" should s"successfully score sentences " in {
      info(dataset.schema.mkString(","))
      AnnotatorBuilder.withPragmaticSentimentDetector(dataset)
        .collect().foreach {
        row =>
          val document = Document(row.getAs[Row](0))
          println(document)
          row.getSeq[Row](4).map(Annotation(_)).foreach {
            matchedAnnotation =>
              println(matchedAnnotation, matchedAnnotation.metadata.mkString(","))
          }
      }
    }
  }

}
