package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.{Annotation, AnnotatorBuilder}
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._
import com.johnsnowlabs.nlp.AnnotatorType.CHUNK
import scala.language.reflectiveCalls

trait RegexMatcherBehaviors { this: FlatSpec =>
  def fixture(dataset: Dataset[Row], rules: Array[(String, String)], strategy: String) = new {
    val annotationDataset: Dataset[_] = AnnotatorBuilder.withRegexMatcher(dataset, strategy)
    val regexAnnotations: Array[Annotation] = annotationDataset.select("regex")
      .collect
      .flatMap { _.getSeq[Row](0) }
      .map { Annotation(_) }

  }

  def customizedRulesRegexMatcher(dataset: => Dataset[Row], rules: Array[(String, String)], strategy: String): Unit = {
    "A RegexMatcher Annotator with custom rules" should s"successfuly match ${rules.map(_._1).mkString(",")}" in {
      val f = fixture(dataset, rules, strategy)
      f.regexAnnotations.foreach { a =>
        assert(Seq("followed by 'the'", "ceremony").contains(a.metadata("identifier")))
      }
    }

    it should "create annotations" in {
      val f = fixture(dataset, rules, strategy)
      assert(f.regexAnnotations.nonEmpty)
    }

    it should "create annotations with the correct tag" in {
      val f = fixture(dataset, rules, strategy)
      f.regexAnnotations.foreach { a =>
        assert(a.annotatorType == CHUNK)
      }
    }
  }
}