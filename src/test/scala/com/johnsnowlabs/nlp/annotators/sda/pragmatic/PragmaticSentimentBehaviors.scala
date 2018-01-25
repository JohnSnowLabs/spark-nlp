package com.johnsnowlabs.nlp.annotators.sda.pragmatic

import com.johnsnowlabs.nlp.annotators.common.TokenizedSentence
import com.johnsnowlabs.nlp.{Annotation, AnnotatorBuilder}
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._
import com.johnsnowlabs.nlp.AnnotatorType.SENTIMENT
import scala.language.reflectiveCalls

trait PragmaticSentimentBehaviors { this: FlatSpec =>

  def fixture(dataset: Dataset[Row]) = new {
    val df = AnnotatorBuilder.withPragmaticSentimentDetector(dataset)
    val sdAnnotations = Annotation.collect(df, "sentiment").flatten
  }

  def isolatedSentimentDetector(tokenizedSentences: Array[TokenizedSentence], expectedScore: Double): Unit = {
    s"tagged sentences" should s"have an expected score of $expectedScore" in {
      val pragmaticScorer = new PragmaticScorer(SentimentDetector.retrieveSentimentDict("/sentiment-corpus/default-sentiment-dict.txt", "txt", ","))
      val result = pragmaticScorer.score(tokenizedSentences)
      assert(result == expectedScore, s"because result: $result did not match expected: $expectedScore")
    }
  }

  def sparkBasedSentimentDetector(dataset: => Dataset[Row]): Unit = {

    "A Pragmatic Sentiment Analysis Annotator" should s"create annotations" in {
      val f = fixture(dataset)
      assert(f.sdAnnotations.size > 0)
    }

    it should "create annotations with the correct type" in {
      val f = fixture(dataset)
      f.sdAnnotations.foreach { a =>
        assert(a.annotatorType == SENTIMENT)
      }
    }

    it should "successfully score sentences" in {
      val f = fixture(dataset)
      f.sdAnnotations.foreach { a =>
        assert(List("positive", "negative").contains(a.result))
      }
    }
  }
}