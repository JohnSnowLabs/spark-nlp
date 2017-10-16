package com.johnsnowlabs.nlp.annotators

import java.util.Date

import com.johnsnowlabs.nlp.{Annotation, AnnotatorBuilder}
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._
import org.scalatest.Matchers._
import com.johnsnowlabs.nlp.AnnotatorType.DATE
import scala.language.reflectiveCalls

trait DateMatcherBehaviors extends FlatSpec {
  def fixture(dataset: Dataset[Row]) = new {
    val df = AnnotatorBuilder.withDateMatcher(dataset)
    val dateAnnotations = df.select("date")
      .collect
      .flatMap { _.getSeq[Row](0) }
      .map { Annotation(_) }
  }

  def sparkBasedDateMatcher(dataset: => Dataset[Row]): Unit = {
    "A DateMatcher Annotator" should s"successfuly parse dates" in {
      val f = fixture(dataset)
      f.dateAnnotations.foreach { a =>
        val d: String = a.result
        d should fullyMatch regex """\d+/\d+/\d+"""
      }
    }

    it should "create annotations" in {
      val f = fixture(dataset)
      assert(f.dateAnnotations.size > 0)
    }

    it should "create annotations with the correct type" in {
      val f = fixture(dataset)
      f.dateAnnotations.foreach { a =>
        assert(a.annotatorType == DATE)
      }
    }
  }
}
