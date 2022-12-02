/*
 * Copyright 2017-2022 John Snow Labs
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

package com.johnsnowlabs.nlp.annotators.btm

import com.johnsnowlabs.nlp.{Annotation, AnnotatorBuilder}
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest.flatspec.AnyFlatSpec

trait BigTextMatcherBehaviors { this: AnyFlatSpec =>

  def fullBigTextMatcher(dataset: => Dataset[Row]) {
    "An BigTextMatcher Annotator" should "successfully transform data" taggedAs FastTest in {
      AnnotatorBuilder
        .withFullBigTextMatcher(dataset)
        .collect()
        .foreach { row =>
          row
            .getSeq[Row](3)
            .map(Annotation(_))
            .foreach {
              case entity: Annotation if entity.annotatorType == "entity" =>
                println(entity, entity.end)
              case _ => ()
            }
        }
    }
  }
}
