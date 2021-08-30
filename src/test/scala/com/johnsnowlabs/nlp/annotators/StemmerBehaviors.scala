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

import com.johnsnowlabs.nlp.{Annotation, AnnotatorBuilder, AnnotatorType}
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest.flatspec.AnyFlatSpec

trait StemmerBehaviors { this: AnyFlatSpec =>

  def fullStemmerPipeline(dataset: => Dataset[Row]) {
    "A Stemmer Annotator" should "successfully transform data" taggedAs FastTest in {
      AnnotatorBuilder.withFullStemmer(dataset)
        .collect.foreach {
        row =>
          row.getSeq[Row](2)
            .map(Annotation(_))
            .foreach {
              case stem: Annotation if stem.annotatorType == AnnotatorType.TOKEN =>
                println(stem, stem.result)
              case _ => ()
            }
      }
    }
  }
}
