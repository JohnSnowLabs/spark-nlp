package com.jsl.nlp.annotators

import com.jsl.nlp.{Annotation, AnnotatorBuilder}
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._
import com.jsl.nlp.AnnotatorType.REGEX
import scala.language.reflectiveCalls

trait RegexMatcherBehaviors { this: FlatSpec =>
  def fixture(dataset: Dataset[Row], rules: Array[(String, String)], strategy: String) = new {
    val df = AnnotatorBuilder.withRegexMatcher(dataset, rules, strategy)
    val regexAnnotations = df.select("regex")
      .collect
      .flatMap { _.getSeq[Row](0) }
      .map { Annotation(_) }

    df.show
  }

  def predefinedRulesRegexMatcher(dataset: => Dataset[Row], rules: Array[(String, String)], strategy: String): Unit = {
//    "A RegexMatcher Annotator" should s"successfuly match ${rules.map(_._1).mkString(",")}" in {
//      val f = fixture(dataset, rules, strategy)
//      f.regexAnnotations.foreach { a =>
//        assert(a.metadata == REGEX)
//      }
//    }
//
//    it should "create annotations" in {
//      val f = fixture(dataset, rules, strategy)
//      assert(f.regexAnnotations.size > 0)
//    }
//
//    it should "create annotations with the correct tag" in {
//      val f = fixture(dataset, rules, strategy)
//      f.regexAnnotations.foreach { a =>
//        assert(a.annotatorType == REGEX)
//      }
//    }
  }
}