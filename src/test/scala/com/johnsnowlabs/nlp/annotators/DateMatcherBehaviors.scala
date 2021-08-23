/*
 * Copyright 2017-2021 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorType.DATE
import com.johnsnowlabs.nlp.{Annotation, AnnotatorBuilder}
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest.Matchers._
import org.scalatest._

import scala.language.reflectiveCalls

trait DateMatcherBehaviors extends FlatSpec {
  def fixture(dataset: Dataset[Row]) = new {
    val df: Dataset[Row] = AnnotatorBuilder.withDateMatcher(dataset)
    val dateAnnotations: Array[Annotation] = df.select("date")
      .collect
      .flatMap {
        _.getSeq[Row](0)
      }
      .map {
        Annotation(_)
      }
  }

  def sparkBasedDateMatcher(dataset: => Dataset[Row]): Unit = {
    "A DateMatcher Annotator" should s"successfuly parse dates" taggedAs FastTest in {
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
