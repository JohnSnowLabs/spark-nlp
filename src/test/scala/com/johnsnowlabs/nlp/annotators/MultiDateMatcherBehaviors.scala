package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorType.DATE
import com.johnsnowlabs.nlp.{Annotation, AnnotatorBuilder}
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest.Matchers._
import org.scalatest._

import scala.language.reflectiveCalls

trait MultiDateMatcherBehaviors extends FlatSpec {
  def fixture(dataset: Dataset[Row]) = new {
    val df = AnnotatorBuilder.withMultiDateMatcher(dataset)
    val dateAnnotations = df.select("date")
      .collect
      .flatMap { _.getSeq[Row](0) }
      .map { Annotation(_) }
  }

  def sparkBasedDateMatcher(dataset: => Dataset[Row]): Unit = {
    "A MultiDateMatcher Annotator" should s"successfuly parse dates" taggedAs FastTest in {
      val f = fixture(dataset)
      f.dateAnnotations.foreach { a =>
        val d: String = a.result
        d should fullyMatch regex """\d+/\d+/\d+"""
      }
    }

    it should "create annotations" taggedAs FastTest in {
      val f = fixture(dataset)
      assert(f.dateAnnotations.size > 0)
    }

    it should "create annotations with the correct type" taggedAs FastTest in {
      val f = fixture(dataset)
      f.dateAnnotations.foreach { a =>
        assert(a.annotatorType == DATE)
      }
    }
  }
}
